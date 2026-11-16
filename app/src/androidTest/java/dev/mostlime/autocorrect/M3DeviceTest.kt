package dev.mostlime.autocorrect

import ai.onnxruntime.OrtEnvironment
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.mostlime.autocorrect.engine.DeviceCpu
import dev.mostlime.autocorrect.engine.Lexicon
import dev.mostlime.autocorrect.engine.SPTokenizer
import dev.mostlime.autocorrect.engine.m3.M3CandGen
import dev.mostlime.autocorrect.engine.m3.M3Dict
import dev.mostlime.autocorrect.engine.m3.M3Ime
import dev.mostlime.autocorrect.engine.m3.M3Lexicon
import dev.mostlime.autocorrect.engine.m3.M3Model
import dev.mostlime.autocorrect.engine.m3.M3Words
import dev.mostlime.autocorrect.engine.m3.QuantTrigram
import dev.mostlime.autocorrect.engine.m3.SymSpellIndex
import dev.mostlime.autocorrect.model.Manifest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.nio.channels.FileChannel
import java.security.MessageDigest

/**
 * m3 on the device / emulator: downloads the keylime-m3-preview files named in the bundled manifest (sha-checked),
 * runs the Python test vectors through onnxruntime-android (texts + tiers must match), and measures latency:
 * a whole message, and the keyboard's word-end call (sentence so far, last two slots rescored). Logged as "AcM3Bench".
 */
@RunWith(AndroidJUnit4::class)
class M3DeviceTest {
    private val inst = InstrumentationRegistry.getInstrumentation()
    private val ctx = inst.targetContext

    private fun fetch(url: String, dst: File, size: Long, sha: String): Boolean {
        if (dst.exists() && dst.length() == size) return true
        val c = (URL(url).openConnection() as HttpURLConnection).apply { connectTimeout = 20_000; readTimeout = 60_000 }
        try {
            if (c.responseCode != 200) return false
            val md = MessageDigest.getInstance("SHA-256")
            val tmp = File(dst.path + ".part")
            c.inputStream.use { inp -> tmp.outputStream().use { out ->
                val buf = ByteArray(1 shl 16)
                while (true) { val n = inp.read(buf); if (n < 0) break; out.write(buf, 0, n); md.update(buf, 0, n) }
            } }
            if (md.digest().joinToString("") { "%02x".format(it) } != sha) { tmp.delete(); return false }
            return tmp.renameTo(dst)
        } catch (e: Exception) {
            Log.w(TAG, "download $url: $e"); return false
        } finally { c.disconnect() }
    }

    private fun mmap(f: File) = RandomAccessFile(f, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }

    @Test fun vectorsAndLatency() {
        val man = Manifest.parse(ctx.assets.open("manifest.json").bufferedReader().use { it.readText() })
        val spec = man.model("m3")!!
        val info = spec.m3!!
        val dir = File(ctx.cacheDir, "m3test").apply { mkdirs() }
        val ok = spec.files.all { fetch(man.mirrors(spec).first() + it.name, File(dir, it.name), it.size, it.sha256) }
        assumeTrue("m3 files not downloadable", ok)
        fun f(r: dev.mostlime.autocorrect.model.RemoteFile) = File(dir, r.name)
        val t0 = System.nanoTime()
        val L = M3Lexicon(Lexicon(f(info.lexicon).readText()))
        val gen = M3CandGen(L, SymSpellIndex(mmap(f(info.symspell))))
        val lm = info.ngram?.let { QuantTrigram(mmap(f(it))) }
        val tok = SPTokenizer.fromJson(f(info.vocab).readText())
        val tLex = (System.nanoTime() - t0) / 1e6
        val threads = DeviceCpu.intraOpThreads()
        val t1 = System.nanoTime()
        val model = M3Model.open(OrtEnvironment.getEnvironment(), mmap(f(info.encoder)), mmap(f(info.decoder)), threads)
        val tSess = (System.nanoTime() - t1) / 1e6
        val C = M3Dict(model, tok, L, gen, lm, info.params)
        Log.i(TAG, "load: data ${"%.0f".format(tLex)} ms, sessions ${"%.0f".format(tSess)} ms, threads $threads [${DeviceCpu.describe()}]")
        val ref = Json.parseToJsonElement(inst.context.assets.open("m3_test_vectors.json").bufferedReader().use { it.readText() }).jsonObject
        C.correct("warm up the model", "standard")
        var bad = 0
        val msgMs = ArrayList<Double>()
        val wordMs = ArrayList<Double>()
        var words = 0
        for (v0 in ref["vectors"]!!.jsonArray) {
            val v = v0.jsonObject
            val input = v["input"]!!.jsonPrimitive.content
            val style = v["style"]!!.jsonPrimitive.content
            val r = C.correct(input, style)
            msgMs.add(r.ms); words += M3Words.SEG.findAll(input).count()
            val auto = v["auto"]!!.jsonArray.map { it.jsonObject["typed"]!!.jsonPrimitive.content + "->" + it.jsonObject["to"]!!.jsonPrimitive.content }
            val sugg = v["suggest"]!!.jsonArray.map { it.jsonObject["typed"]!!.jsonPrimitive.content + "->" + it.jsonObject["to"]!!.jsonPrimitive.content }
            if (r.greedy != v["constrained_greedy"]!!.jsonPrimitive.content || r.text != v["auto_text"]!!.jsonPrimitive.content ||
                r.auto.map { it.typed + "->" + it.to } != auto || r.suggest.map { it.typed + "->" + it.to } != sugg) {
                bad++; Log.w(TAG, "mismatch [$style] $input -> ${r.text} | auto ${r.auto.map { it.to }} sugg ${r.suggest.map { it.to }}")
            }
            // the keyboard: a call at every word end with the sentence so far, deciding the last two words
            if (style == "standard") {
                val ms = M3Words.SEG.findAll(input).toList()
                for (k in ms.indices) {
                    val win = M3Ime.window(input.substring(0, ms[k].range.last + 1))
                    val n = M3Words.SEG.findAll(win).count()
                    wordMs.add(C.correct(win, style, slots = setOf(n - 2, n - 1)).ms)
                }
            }
        }
        model.close()
        fun pct(a: List<Double>, q: Double) = a.sorted()[((a.size - 1) * q).toInt()]
        Log.i(TAG, "vectors: ${msgMs.size} messages, $bad mismatches")
        Log.i(TAG, "whole message: p50 ${"%.0f".format(pct(msgMs, 0.5))} ms, p90 ${"%.0f".format(pct(msgMs, 0.9))} ms, ${"%.1f".format(msgMs.sum() / words)} ms/word")
        Log.i(TAG, "word-end call: n ${wordMs.size}, p50 ${"%.0f".format(pct(wordMs, 0.5))} ms, p90 ${"%.0f".format(pct(wordMs, 0.9))} ms, max ${"%.0f".format(wordMs.max())} ms")
        Log.i(TAG, "time split (ms total): encoder ${"%.0f".format(C.timing[0])}, greedy ${"%.0f".format(C.timing[1])}, cands ${"%.0f".format(C.timing[2])}, rescoring ${"%.0f".format(C.timing[3])}")
        val rt = Runtime.getRuntime()
        Log.i(TAG, "java heap used ${(rt.totalMemory() - rt.freeMemory()) / 1_000_000} MB, native ${android.os.Debug.getNativeHeapAllocatedSize() / 1_000_000} MB")
        assertTrue("$bad vector mismatches on device", bad == 0)
    }

    companion object { const val TAG = "AcM3Bench" }
}
