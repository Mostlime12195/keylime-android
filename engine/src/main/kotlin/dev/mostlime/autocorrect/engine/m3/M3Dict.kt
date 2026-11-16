package dev.mostlime.autocorrect.engine.m3

import dev.mostlime.autocorrect.engine.SPTokenizer
import dev.mostlime.autocorrect.engine.SpatialModel
import dev.mostlime.autocorrect.engine.KeyGeom

/** One change the decision made: typed word -> alternative, with its margin over keeping the typed word. */
class M3Change(val j: Int, val typed: String, val to: String, val margin: Double, val span: IntArray)

class M3Result(
    /** greedy text with every AUTO change applied (the others reverted to the typed words) */
    val text: String,
    /** constrained greedy output */
    val greedy: String,
    val auto: List<M3Change>,
    val suggest: List<M3Change>,
    /** no slot needed rescoring */
    val fast: Boolean,
    /** greedy steps where the constraint overrode the argmax */
    val blocked: Int,
    val plans: List<M3Plan>,
    val ms: Double,
)

/**
 * m3 + dictionary-aware decoding (m3dict_cpt/PORT_SPEC.md; reference onnx_backend.M3Dict.correct):
 * constrained greedy -> align + plan uncertain slots -> candidates -> rescoring (m3 deltas + spatial + trigram + prior)
 * -> AUTO (margin >= th_auto) / SUGGEST (margin >= th_sugg) per slot. Deterministic: ties (score desc, id asc).
 */
class M3Dict(
    val model: M3Model,
    val tok: SPTokenizer,
    val L: M3Lexicon,
    val gen: M3CandGen,
    val lm: QuantTrigram?,
    val P: M3Params,
    val styleIds: Map<String, Int> = mapOf("casual" to 256, "standard" to 257, "formal" to 258),
    val maxSrc: Int = 384,
    val maxTgt: Int = 112,
    val topK: Int = 8,
    val tauSplit: Double = 4.0,
    /** RARE of kprep (rank from which a lexicon word gets candidates when m3 copied it with low confidence) */
    val rareRank: Int = 10000,
    val eos: Int = 2,
) {
    private val ptext: Array<String?> = Array(tok.vocabSize) { i ->
        val p = tok.pieces[i]
        if (p.startsWith("<0x") && p.endsWith(">") && p.length == 6) {
            val b = p.substring(3, 5).toInt(16); if (b < 128) b.toChar().toString() else "\u0080"
        } else if (tok.types[i] == SPTokenizer.T_CONTROL || tok.types[i] == SPTokenizer.T_UNKNOWN) null
        else p.replace(SPTokenizer.WS, " ")
    }
    private val spm = SpatialModel()
    /** cumulative timings (ms): encoder, constrained greedy, candidates + plan, rescoring (debug / bench) */
    val timing = DoubleArray(4)
    private fun lap(i: Int, t: Long): Long { val n = System.nanoTime(); timing[i] += (n - t) / 1e6; return n }
    private val floor = L.minLogp - 1.0

    fun srcIds(text: String, style: String): IntArray {
        val b = text.toByteArray(Charsets.UTF_8)
        val n = minOf(maxSrc, b.size + 1)
        return IntArray(n) { if (it == 0) styleIds[style] ?: styleIds["standard"]!! else b[it - 1].toInt() and 0xff }
    }

    private class Top(val ids: IntArray, val lps: FloatArray)

    /** top-k of lp by (lp desc, id asc) */
    private fun topK(lp: FloatArray, k: Int): IntArray {
        val ids = IntArray(k) { -1 }
        val v = FloatArray(k) { Float.NEGATIVE_INFINITY }
        var cnt = 0
        for (t in lp.indices) {
            val x = lp[t]
            if (cnt == k && !(x > v[k - 1])) continue
            var p = if (cnt < k) cnt++ else k - 1
            while (p > 0 && x > v[p - 1]) { v[p] = v[p - 1]; ids[p] = ids[p - 1]; p-- }
            v[p] = x; ids[p] = t
        }
        return ids.copyOf(cnt)
    }

    class Greedy(val toks: IntArray, val blocked: Int, val lps: DoubleArray)

    fun constrainedGreedy(src: IntArray, e: M3Model.Enc, ctx: M3Ctx): Greedy {
        var state = model.fresh()
        var cs = M3State.START
        val out = ArrayList<Int>()
        val lps = ArrayList<Double>()
        var nblk = 0
        var ids = intArrayOf(eos)
        val steps = minOf(maxTgt, src.size / 2 + 24)
        for (step in 0 until steps) {
            val r = model.run(e, state, ids)
            state = r.state
            val lp = r.lastLp
            val top = topK(lp, topK)
            var cand: List<Int> = top.toList()
            var choiceTok = -1
            var choiceSt: M3State? = null
            var splitTok = -1
            var splitSt: M3State? = null
            var splitV = 0f
            var scanned = false
            while (choiceSt == null) {
                for (t in cand) {
                    val v = lp[t]
                    if (t == eos) {
                        if (M3Feed.finishOk(ctx, cs)) { choiceTok = t; choiceSt = cs; break }
                        continue
                    }
                    val pt = ptext[t] ?: continue
                    val f = M3Feed.feed(ctx, cs, pt)
                    if (!f.ok) continue
                    if (f.split) {
                        if (splitTok < 0) { splitTok = t; splitSt = f.st; splitV = v }
                        continue
                    }
                    choiceTok = t; choiceSt = f.st
                    // Python: float(lp) differences in float64
                    if (splitTok >= 0 && splitV.toDouble() - v.toDouble() >= tauSplit) { choiceTok = splitTok; choiceSt = splitSt }
                    break
                }
                if (choiceSt == null) {
                    if (scanned) {
                        if (splitTok >= 0) { choiceTok = splitTok; choiceSt = splitSt } else { choiceTok = eos; choiceSt = cs }
                        break
                    }
                    val order = lp.indices.sortedWith { a, b -> if (lp[a] != lp[b]) lp[b].compareTo(lp[a]) else a.compareTo(b) }
                    cand = order.subList(minOf(topK, order.size), order.size)
                    scanned = true
                }
            }
            cs = choiceSt!!
            if (choiceTok != top[0]) nblk++
            if (choiceTok == eos) break
            out.add(choiceTok); ids = intArrayOf(choiceTok); lps.add(lp[choiceTok].toDouble())
        }
        return Greedy(out.toIntArray(), nblk, lps.toDoubleArray())
    }

    /** log P(v + EOS) - log P(base + EOS) per variant; the decoder state after the common prefix is shared */
    fun deltaScores(e: M3Model.Enc, base: IntArray, variants: List<IntArray>,
                    cache: HashMap<Int, Pair<M3Model.State, Double>> = HashMap()): DoubleArray {
        return DoubleArray(variants.size) { vi ->
            val v = variants[vi]
            var Lc = 0
            while (Lc < minOf(v.size, base.size) && v[Lc] == base[Lc]) Lc++
            val pre = IntArray(Lc + 1) { if (it == 0) eos else base[it - 1] }
            val (st, bl) = cache.getOrPut(Lc) {
                val s0 = if (Lc == 0) model.fresh() else model.run(e, model.fresh(), pre.copyOf(pre.size - 1), wantLast = false).state
                val tl = model.run(e, s0, intArrayOf(pre.last()) + base.copyOfRange(Lc, base.size), base.copyOfRange(Lc, base.size) + eos, wantLast = false).tokLp
                s0 to M3Model.npSum(tl).toDouble()
            }
            val tl = model.run(e, st, intArrayOf(pre.last()) + v.copyOfRange(Lc, v.size), v.copyOfRange(Lc, v.size) + eos, wantLast = false).tokLp
            M3Model.npSum(tl).toDouble() - bl
        }
    }

    /** sum of greedy token log-probs over each output SEG word (code-point offsets, as the reference) */
    fun wordLps(out: String, toks: IntArray, lps: DoubleArray): DoubleArray {
        val full = StringBuilder()
        for (t in toks) full.append(ptext[t] ?: "")
        val f = full.toString()
        val lead = cpLen(f) - cpLen(f.trimStart(' '))
        val spans = ArrayList<IntArray>()
        var pos = 0
        for (t in toks) { val n = cpLen(ptext[t] ?: ""); spans.add(intArrayOf(pos - lead, pos + n - lead)); pos += n }
        return M3Words.SEG.findAll(out).map { m ->
            val a = out.codePointCount(0, m.range.first)
            val b = out.codePointCount(0, m.range.last + 1)
            var s = 0.0
            for (i in spans.indices) if (spans[i][0] < b && spans[i][1] > a) s += lps[i]
            s
        }.toList().toDoubleArray()
    }

    private fun cpLen(s: String) = s.codePointCount(0, s.length)

    private fun cands(w: String, cache: HashMap<String, List<M3Cand>>) = cache.getOrPut(w) { gen.cands(w, k = 8) }

    /** decode.slot_plan */
    fun slotPlan(typed: String, out: String, wlp: DoubleArray?, lowconf: Double?, rare: Int, cache: HashMap<String, List<M3Cand>>, k: Int = 8): List<M3Plan> {
        val tw = M3Words.SEG.findAll(typed).map { it.value }.toList()
        val owm = M3Words.SEG.findAll(out).toList()
        val asg = M3Align.alignIdx(tw, owm.map { it.value })
        val prot = M3Words.protectedSlots(typed)
        val plans = ArrayList<M3Plan>()
        for ((j, t) in tw.withIndex()) {
            val idx = asg[j]
            if (idx.isEmpty() || idx != (idx[0]..idx.last()).toList()) continue
            val o = idx.joinToString(" ") { owm[it].value }
            val tl = t.lowercase()
            val pj = prot.getOrElse(j) { false }
            val nonword = !pj && !(L.has(tl) || (tl.endsWith("'s") && L.has(tl.dropLast(2))))
            val changed = o.lowercase() != tl
            val rank = L.id(tl).takeIf { it >= 0 } ?: (if (tl.endsWith("'s")) L.id(tl.dropLast(2)) else -1)
            var rareF = !pj && !nonword && rank >= rare && tl.length >= 3
            val lpW = if (wlp != null && idx.last() < wlp.size) idx.sumOf { wlp[it] } else 0.0
            val low = lowconf != null && !pj && !nonword && !changed && lpW < lowconf && tl.length >= 2
            rareF = rareF || low
            if (!(nonword || changed || rareF)) continue
            val pat = owm[idx[0]].value
            val copy = if (pat.take(1).let { it.isNotEmpty() && it[0].isUpperCase() } && t.take(1).let { it.isNotEmpty() && it[0].isLowerCase() }) t.take(1).uppercase() + t.drop(1) else t
            val alts = ArrayList<M3Alt>()
            alts.add(M3Alt(copy, "copy"))
            if (changed) alts.add(M3Alt(o, "greedy"))
            if (nonword || rareF) {
                val cl = cands(tl, cache)
                val have = alts.mapTo(HashSet()) { it.text.lowercase() }
                for (c in cl.take(k)) {
                    val ct = M3Words.applyCase(c.text, pat)
                    if (ct.lowercase() !in have) { alts.add(M3Alt(ct, c.kind)); have.add(ct.lowercase()) }
                }
            }
            val nameLike = (t.take(1).let { it.isNotEmpty() && it[0].isUpperCase() } && j > 0 && t != "I") || (t.length > 1 && M3Words.isUpper(t))
            plans.add(M3Plan(j, t, intArrayOf(owm[idx[0]].range.first, owm[idx.last()].range.last + 1), o, nonword, pj, changed, rareF, low,
                M3Words.round(lpW, 4), rank, nameLike, alts))
        }
        return plans
    }

    /** spatial log-lik of the typed word given an alternative (letters + apostrophes only) */
    private fun spFeat(typed: String, alt: String): Double {
        val t = typed.lowercase().filter { it.isLetter() || it == '\'' }
        val a = alt.lowercase().filter { it in 'a'..'z' || it == '\'' }
        if (a.isEmpty()) return -50.0
        val code = spm.wordCode(a) ?: return -50.0
        return spm.score(spm.emitTable(KeyGeom.keyTouches(t)), code)
    }

    private fun prFeat(alt: String): Double {
        var s = 0.0
        for (m in M3Words.SEG.findAll(alt)) { val i = L.id(m.value.lowercase()); s += if (i >= 0) L.logp[i] else floor }
        return s
    }

    private fun sentLp(text: String) = lm!!.sentLp(M3Words.SEG.findAll(text).map { it.value.lowercase() }.toList())

    fun encodeText(s: String): IntArray = tok.encode(s.replace("\n", "\n "))

    /**
     * slots: only these typed-word indices are rescored and decided (the keyboard's word-end call needs the last two);
     * null = all (the reference). Every slot is decided on its own, so a subset gets the same decisions.
     */
    fun correct(text: String, style: String = "standard", user: Collection<String> = emptyList(), slots: Set<Int>? = null): M3Result {
        val t0 = System.nanoTime()
        val src = srcIds(text, style)
        var tl = System.nanoTime()
        model.encode(src).use { e ->
            tl = lap(0, tl)
            val cache = HashMap<String, List<M3Cand>>()
            val inj = ArrayList<String>()
            for (m in M3Words.SEG.findAll(text)) {
                val wl = m.value.lowercase()
                if (!L.has(wl)) for (c in cands(wl, cache)) inj.add(c.text)
            }
            tl = lap(2, tl)
            val g = constrainedGreedy(src, e, M3Ctx(L, text, user, inj))
            tl = lap(1, tl)
            val o = tok.decode(g.toks).replace("\n ", "\n")
            val lc = P.lowconf
            var plans = slotPlan(text, o, wordLps(o, g.toks, g.lps), if (lc > -1e8) lc else null, minOf(rareRank.toDouble(), P.rare).toInt(), cache)
            if (slots != null) plans = plans.filter { it.j in slots }
            // decoder states / base scores after a shared prefix: the same calls for every slot of the message
            val prefixCache = HashMap<Int, Pair<M3Model.State, Double>>()
            tl = lap(2, tl)
            val base = encodeText(o)
            val baseLp = if (lm != null && P.b != 0.0) sentLp(o) else 0.0
            var fast = true
            for (p in plans) {
                val keep = M3Decide.decideAlts(p, P)
                var ds: Iterator<Double>? = null
                if (keep.size > 1) {
                    fast = false
                    val vs = keep.filter { p.alts[it].kind != "greedy" }.map { encodeText(M3Words.variant(o, p.span, p.alts[it].text)) }
                    ds = deltaScores(e, base, vs, prefixCache).iterator()
                }
                for ((i, a) in p.alts.withIndex()) {
                    val dm3 = if (i !in keep || keep.size == 1) (if (a.kind != "copy" || !p.changed) 0.0 else -1e9)
                        else if (a.kind == "greedy") 0.0 else ds!!.next()
                    var ng = 0.0
                    if (lm != null && P.b != 0.0 && a.kind != "greedy") ng = sentLp(M3Words.variant(o, p.span, a.text)) - baseLp
                    a.dm3 = M3Words.round(dm3, 4); a.sp = M3Words.round(spFeat(p.typed, a.text), 4)
                    a.pr = M3Words.round(prFeat(a.text), 4); a.ng = M3Words.round(ng, 4)
                }
            }
            tl = lap(3, tl)
            if (style == "casual") plans = M3Decide.casualFilter(plans)
            val auto = ArrayList<M3Change>()
            val sugg = ArrayList<M3Change>()
            for (p in plans) {
                val d = M3Decide.decide(p, P)
                if (d.kind == "copy") continue
                val c = M3Change(p.j, p.typed, d.text, M3Words.round(d.margin, 3), p.span)
                if (d.margin >= P.thAuto) auto.add(c) else if (d.margin >= P.thSugg) sugg.add(c)
            }
            return M3Result(M3Decide.apply(o, plans, P, P.thAuto), o, auto, sugg, fast, g.blocked, plans, (System.nanoTime() - t0) / 1e6)
        }
    }
}
