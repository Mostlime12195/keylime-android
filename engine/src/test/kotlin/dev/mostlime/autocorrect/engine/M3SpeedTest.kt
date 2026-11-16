package dev.mostlime.autocorrect.engine

import ai.onnxruntime.OrtEnvironment
import dev.mostlime.autocorrect.engine.m3.M3CandGen
import dev.mostlime.autocorrect.engine.m3.M3Dict
import dev.mostlime.autocorrect.engine.m3.M3Ime
import dev.mostlime.autocorrect.engine.m3.M3Lexicon
import dev.mostlime.autocorrect.engine.m3.M3Model
import dev.mostlime.autocorrect.engine.m3.M3Params
import dev.mostlime.autocorrect.engine.m3.M3Words
import dev.mostlime.autocorrect.engine.m3.QuantTrigram
import dev.mostlime.autocorrect.engine.m3.SymSpellIndex
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.io.RandomAccessFile
import java.nio.channels.FileChannel

/** Word-end latency of the keyboard path (sentence so far, last two slots) on the vector sentences. AC_M3_BENCH=1. */
class M3SpeedTest {
    private val dir = System.getenv("AC_MODEL_DIR").orEmpty()
    private fun f(name: String) = File(dir, name)
    private fun mmap(file: File) = RandomAccessFile(file, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }

    @Test fun wordEndLatency() {
        assumeTrue(System.getenv("AC_M3_BENCH") == "1" && f("m3.decoder.int4.onnx").exists())
        val ref = Fixtures.json("m3_test_vectors.json").jsonObject
        val p = ref["params"]!!.jsonObject
        fun d(k: String) = p[k]!!.jsonPrimitive.double
        val P = M3Params(p["k"]!!.jsonPrimitive.int, d("rare"), d("lowconf"), d("a"), d("b"), d("d"), d("c_name"), d("c_real"), d("c_rare"), d("th_auto"), d("th_sugg"))
        val L = M3Lexicon(Lexicon(f("m3_lexicon.tsv").readText()))
        val C = M3Dict(M3Model.open(OrtEnvironment.getEnvironment(), mmap(f("m3.encoder.int4.onnx")), mmap(f("m3.decoder.int4.onnx")), threads = 2),
            Fixtures.tokM2, L, M3CandGen(L, SymSpellIndex(mmap(f("m3_symspell.bin")))), QuantTrigram(mmap(f("m3_ngram.bin"))), P)
        C.correct("warm up the model", "standard")
        val ms = ArrayList<Double>()
        for (v in ref["vectors"]!!.jsonArray) {
            val input = v.jsonObject["input"]!!.jsonPrimitive.content
            if (v.jsonObject["style"]!!.jsonPrimitive.content != "standard") continue
            for (m in M3Words.SEG.findAll(input)) {
                val win = M3Ime.window(input.substring(0, m.range.last + 1))
                val n = M3Words.SEG.findAll(win).count()
                ms.add(C.correct(win, "standard", slots = setOf(n - 2, n - 1)).ms)
            }
        }
        val s = ms.sorted()
        println("M3SPEED word-end: n ${s.size} p50 %.0f p90 %.0f max %.0f ms; split enc %.0f greedy %.0f cands %.0f rescoring %.0f".format(
            s[s.size / 2], s[(s.size * 9) / 10], s.last(), C.timing[0], C.timing[1], C.timing[2], C.timing[3]))
    }
}
