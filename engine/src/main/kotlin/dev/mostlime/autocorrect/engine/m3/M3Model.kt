package dev.mostlime.autocorrect.engine.m3

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * m3 int4 encoder-decoder through ONNX Runtime, batch 1, KV cache (m3dict onnx_backend.OrtDict).
 * Encoder: src_ids [1,S] (style id + UTF-8 bytes), taps [1,S,2] (zeros) -> cross_kN, cross_vN (precomputed cross K/V).
 * Decoder: ids, next_ids, pos, past_mask, src_mask, past states, cross K/V -> last_lp [1,V] (log-probs after the last id),
 * tok_lp [1,N] (log P(next_ids[i]) teacher forced), present states.
 */
class M3Model private constructor(
    private val env: OrtEnvironment,
    private val enc: OrtSession,
    private val dec: OrtSession,
    val nKvHeads: Int, val headDim: Int, val convKernel: Int, val dModel: Int,
) : AutoCloseable {
    /** past state names in the decoder's input order (k0, v0, c0, c1) */
    val stateNames: List<String> = dec.inputNames.filter { it.startsWith("past_") && it != "past_mask" }.map { it.substring(5) }
    private val crossNames: List<String> = dec.inputNames.filter { it.startsWith("cross_") }
    @Volatile var closed = false
        private set

    /** encoder output for one message: the cross K/V tensors, reused by every decoder call (close() after the message) */
    inner class Enc(val S: Int, val cross: Map<String, OnnxTensor>, private val srcMask: OnnxTensor) : AutoCloseable {
        fun mask() = srcMask
        override fun close() { cross.values.forEach { it.close() }; srcMask.close() }
    }

    class State(val data: Map<String, FloatArray>, val shapes: Map<String, LongArray>, val mask: BooleanArray, val pos: Int)

    class Step(val lastLp: FloatArray, val tokLp: FloatArray, val state: State)

    fun encode(src: IntArray): Enc {
        val S = src.size
        OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(S) { src[it].toLong() }), longArrayOf(1, S.toLong())).use { ids ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(S * 2)), longArrayOf(1, S.toLong(), 2)).use { taps ->
                enc.run(mapOf("src_ids" to ids, "taps" to taps)).use { r ->
                    val cross = HashMap<String, OnnxTensor>()
                    for (nm in crossNames) {
                        val t = r.get(nm).get() as OnnxTensor
                        val fb = t.floatBuffer
                        val arr = FloatArray(fb.remaining()).also { fb.get(it) }
                        cross[nm] = OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), t.info.shape)
                    }
                    val sm = OnnxTensor.createTensor(env, BooleanArray(S) { true })
                    return Enc(S, cross, sm)
                }
            }
        }
    }

    fun fresh(): State {
        val d = HashMap<String, FloatArray>()
        val sh = HashMap<String, LongArray>()
        for (s in stateNames) {
            if (s[0] == 'k' || s[0] == 'v') { sh[s] = longArrayOf(1, nKvHeads.toLong(), 1, headDim.toLong()); d[s] = FloatArray(nKvHeads * headDim) }
            else { sh[s] = longArrayOf(1, (convKernel - 1).toLong(), dModel.toLong()); d[s] = FloatArray((convKernel - 1) * dModel) }
        }
        return State(d, sh, BooleanArray(1), 0)
    }

    fun run(e: Enc, st: State, ids: IntArray, nextIds: IntArray? = null, wantLast: Boolean = true): Step {
        check(!closed) { "m3 session closed" }
        val n = ids.size
        val tensors = ArrayList<OnnxTensor>()
        fun reg(t: OnnxTensor) = t.also { tensors.add(it) }
        try {
            val feed = HashMap<String, OnnxTensor>()
            val shape = longArrayOf(1, n.toLong())
            feed["ids"] = reg(OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n) { ids[it].toLong() }), shape))
            feed["next_ids"] = reg(OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n) { (nextIds?.get(it) ?: 0).toLong() }), shape))
            feed["pos"] = reg(OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(n) { (st.pos + it).toLong() }), longArrayOf(n.toLong())))
            feed["past_mask"] = reg(OnnxTensor.createTensor(env, st.mask))
            feed["src_mask"] = e.mask()
            for ((k, v) in e.cross) feed[k] = v
            for (s in stateNames) feed["past_$s"] = reg(OnnxTensor.createTensor(env, FloatBuffer.wrap(st.data[s]!!), st.shapes[s]!!))
            dec.run(feed).use { r ->
                val last = if (wantLast) (r.get("last_lp").get() as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } } else FloatArray(0)
                val tok = (r.get("tok_lp").get() as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } }
                val d = HashMap<String, FloatArray>()
                val sh = HashMap<String, LongArray>()
                for (s in stateNames) {
                    val t = r.get("present_$s").get() as OnnxTensor
                    val b = t.floatBuffer
                    d[s] = FloatArray(b.remaining()).also { b.get(it) }; sh[s] = t.info.shape
                }
                val mask = BooleanArray(st.mask.size + n) { if (it < st.mask.size) st.mask[it] else true }
                return Step(last, tok, State(d, sh, mask, st.pos + n))
            }
        } finally {
            tensors.forEach { it.close() }
        }
    }

    override fun close() {
        if (closed) return
        closed = true
        runCatching { enc.close() }; runCatching { dec.close() }
    }

    companion object {
        /** model files as bytes (mmap): .onnx, optimized at session creation */
        fun open(env: OrtEnvironment, encoder: ByteBuffer, decoder: ByteBuffer, threads: Int = 2,
                 nKvHeads: Int = 2, headDim: Int = 64, convKernel: Int = 7, dModel: Int = 512): M3Model {
            fun opts() = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setInterOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val e = env.createSession(encoder, opts())
            val d = try { env.createSession(decoder, opts()) } catch (t: Throwable) { e.close(); throw t }
            return M3Model(env, e, d, nKvHeads, headDim, convKernel, dModel)
        }

        /** numpy's float32 pairwise sum (np.sum of a float32 array) */
        fun npSum(a: FloatArray, lo: Int = 0, n: Int = a.size): Float {
            if (n < 8) {
                var r = 0f
                for (i in lo until lo + n) r += a[i]
                return r
            }
            if (n <= 128) {
                val r = FloatArray(8) { a[lo + it] }
                var i = 8
                while (i < n - (n % 8)) { for (j in 0 until 8) r[j] += a[lo + i + j]; i += 8 }
                var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
                while (i < n) { res += a[lo + i]; i++ }
                return res
            }
            var n2 = n / 2
            n2 -= n2 % 8
            return npSum(a, lo, n2) + npSum(a, lo + n2, n - n2)
        }
    }
}
