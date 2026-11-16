package dev.mostlime.autocorrect.engine

import ai.onnxruntime.OrtEnvironment
import dev.mostlime.autocorrect.engine.m3.M3CandGen
import dev.mostlime.autocorrect.engine.m3.M3Dict
import dev.mostlime.autocorrect.engine.m3.M3Lexicon
import dev.mostlime.autocorrect.engine.m3.M3Model
import dev.mostlime.autocorrect.engine.m3.M3Params
import dev.mostlime.autocorrect.engine.m3.M3Words
import dev.mostlime.autocorrect.engine.m3.QuantTrigram
import dev.mostlime.autocorrect.engine.m3.SymSpellIndex
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
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
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * m3 + m3dict against the Python reference (m3dict_cpt/test_vectors.json, made by make_vectors.py on the int4 ONNX):
 * constrained greedy text, plans (alternatives), AUTO text and both tiers must match exactly; features to rounding /
 * float tolerance. Needs AC_MODEL_DIR with the keylime-m3-preview files (CI downloads them).
 */
class M3VectorsTest {
    private val dir = System.getenv("AC_MODEL_DIR").orEmpty()
    private fun f(name: String) = File(dir, name)
    private fun mmap(file: File) = RandomAccessFile(file, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }

    private val names = listOf("m3.encoder.int4.onnx", "m3.decoder.int4.onnx", "m3_lexicon.tsv", "m3_symspell.bin", "m3_ngram.bin")

    private fun params(p: JsonObject) = M3Params(
        k = p["k"]!!.jsonPrimitive.int, rare = p["rare"]!!.jsonPrimitive.double, lowconf = p["lowconf"]!!.jsonPrimitive.double,
        a = p["a"]!!.jsonPrimitive.double, b = p["b"]!!.jsonPrimitive.double, d = p["d"]!!.jsonPrimitive.double,
        cName = p["c_name"]!!.jsonPrimitive.double, cReal = p["c_real"]!!.jsonPrimitive.double, cRare = p["c_rare"]!!.jsonPrimitive.double,
        thAuto = p["th_auto"]!!.jsonPrimitive.double, thSugg = p["th_sugg"]!!.jsonPrimitive.double)

    @Test fun matchesPythonVectors() {
        assumeTrue("no m3 files in AC_MODEL_DIR", dir.isNotEmpty() && names.all { f(it).exists() })
        val ref = Fixtures.json("m3_test_vectors.json").jsonObject
        val P = params(ref["params"]!!.jsonObject)
        val t0 = System.nanoTime()
        val L = M3Lexicon(Lexicon(f("m3_lexicon.tsv").readText()))
        val gen = M3CandGen(L, SymSpellIndex(mmap(f("m3_symspell.bin"))))
        val lm = QuantTrigram(mmap(f("m3_ngram.bin")))
        val model = M3Model.open(OrtEnvironment.getEnvironment(), mmap(f("m3.encoder.int4.onnx")), mmap(f("m3.decoder.int4.onnx")), threads = 2)
        println("m3 load ${(System.nanoTime() - t0) / 1e6} ms")
        val C = M3Dict(model, Fixtures.tokM2, L, gen, lm, P)
        val errs = ArrayList<String>()
        var maxDm3 = 0.0
        val times = ArrayList<Double>()
        var words = 0
        val vs = ref["vectors"]!!.jsonArray
        // warm-up (first ORT runs allocate)
        C.correct("warm up the model", "standard")
        for (v0 in vs) {
            val v = v0.jsonObject
            val input = v["input"]!!.jsonPrimitive.content
            val style = v["style"]!!.jsonPrimitive.content
            val r = C.correct(input, style)
            times.add(r.ms); words += M3Words.SEG.findAll(input).count()
            val tag = "[$style] $input"
            fun chk(what: String, exp: Any?, got: Any?) { if (exp != got) errs.add("$tag: $what expected <$exp> got <$got>") }
            chk("greedy", v["constrained_greedy"]!!.jsonPrimitive.content, r.greedy)
            chk("blocked", v["blocked_steps"]!!.jsonPrimitive.int, r.blocked)
            chk("fast", v["fast_path"]!!.jsonPrimitive.boolean, r.fast)
            chk("auto_text", v["auto_text"]!!.jsonPrimitive.content, r.text)
            fun tier(name: String) = v[name]!!.jsonArray.map { it.jsonObject["typed"]!!.jsonPrimitive.content + " -> " + it.jsonObject["to"]!!.jsonPrimitive.content }
            chk("auto", tier("auto"), r.auto.map { it.typed + " -> " + it.to })
            chk("suggest", tier("suggest"), r.suggest.map { it.typed + " -> " + it.to })
            for ((name, list) in listOf("auto" to r.auto, "suggest" to r.suggest)) {
                v[name]!!.jsonArray.zip(list).forEach { (e, g) ->
                    val d = abs(e.jsonObject["margin"]!!.jsonPrimitive.double - g.margin)
                    if (d > 0.02) errs.add("$tag: $name margin ${e.jsonObject["margin"]} vs ${g.margin}")
                }
            }
            val plans = v["plans"]!!.jsonArray
            chk("plans", plans.map { it.jsonObject["typed"]!!.jsonPrimitive.content }, r.plans.map { it.typed })
            for ((pe, pg) in plans.zip(r.plans)) {
                val p = pe.jsonObject
                chk("greedy_word", p["greedy_word"]!!.jsonPrimitive.content, pg.outWord)
                chk("nonword ${pg.typed}", p["nonword"]!!.jsonPrimitive.boolean, pg.nonword)
                chk("name_like ${pg.typed}", p["name_like"]!!.jsonPrimitive.boolean, pg.nameLike)
                val alts = p["alts"]!!.jsonArray.map { it.jsonArray }
                chk("alts ${pg.typed}", alts.map { it[0].jsonPrimitive.content + "/" + it[1].jsonPrimitive.content }, pg.alts.map { it.text + "/" + it.kind })
                for ((ae, ag) in alts.zip(pg.alts)) {
                    val dm3 = abs(ae[2].jsonPrimitive.double - ag.dm3)
                    if (abs(ae[2].jsonPrimitive.double) < 1e8) maxDm3 = maxOf(maxDm3, dm3)
                    if (dm3 > 0.02) errs.add("$tag: dm3 ${ag.text} ${ae[2]} vs ${ag.dm3}")
                    if (abs(ae[3].jsonPrimitive.double - ag.sp) > 2e-4) errs.add("$tag: sp ${ag.text} ${ae[3]} vs ${ag.sp}")
                    if (abs(ae[4].jsonPrimitive.double - ag.pr) > 2e-4) errs.add("$tag: pr ${ag.text} ${ae[4]} vs ${ag.pr}")
                    if (abs(ae[5].jsonPrimitive.double - ag.ng) > 1e-9) errs.add("$tag: ng ${ag.text} ${ae[5]} vs ${ag.ng}")
                }
            }
        }
        model.close()
        val sorted = times.sorted()
        println("m3 vectors: ${vs.size} messages, ${errs.size} mismatches, max |dm3 diff| %.4f; ms/message p50 %.0f p90 %.0f; ms/word %.1f (2 threads)".format(
            maxDm3, sorted[sorted.size / 2], sorted[(sorted.size * 9) / 10], times.sum() / words))
        println("m3 time split (ms total): encoder %.0f, greedy %.0f, candidates+plan %.0f, rescoring %.0f".format(C.timing[0], C.timing[1], C.timing[2], C.timing[3]))
        errs.forEach { println("  $it") }
        assertTrue(errs.isEmpty(), "${errs.size} mismatches vs the Python vectors")
    }

    /** the trigram feature alone: ng of every alternative from the reference greedy text (no model needed) */
    @Test fun trigramMatchesPython() {
        assumeTrue("no m3_ngram.bin in AC_MODEL_DIR", dir.isNotEmpty() && f("m3_ngram.bin").exists())
        val lm = QuantTrigram(mmap(f("m3_ngram.bin")))
        val ref = Fixtures.json("m3_test_vectors.json").jsonObject
        var n = 0
        for (v0 in ref["vectors"]!!.jsonArray) {
            val v = v0.jsonObject
            val o = v["constrained_greedy"]!!.jsonPrimitive.content
            fun sl(s: String) = lm.sentLp(M3Words.SEG.findAll(s).map { it.value.lowercase() }.toList())
            val base = sl(o)
            for (p0 in v["plans"]!!.jsonArray) {
                val p = p0.jsonObject
                val span = p["span"]!!.jsonArray.map { it.jsonPrimitive.int }.toIntArray()
                for (a in p["alts"]!!.jsonArray.map { it.jsonArray }) {
                    if (a[1].jsonPrimitive.content == "greedy") continue
                    val ng = M3Words.round(sl(M3Words.variant(o, span, a[0].jsonPrimitive.content)) - base, 4)
                    assertEquals(a[5].jsonPrimitive.double, ng, 1e-9, "ng of ${a[0]} in $o")
                    n++
                }
            }
        }
        println("trigram: $n alternatives identical")
    }
}
