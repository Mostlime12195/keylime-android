package dev.mostlime.autocorrect.engine.m3

import dev.mostlime.autocorrect.engine.CandidateGenerator
import dev.mostlime.autocorrect.engine.CandidateGeneratorV2
import dev.mostlime.autocorrect.engine.KeyGeom
import dev.mostlime.autocorrect.engine.SpatialModel
import kotlin.math.abs

/** A generator candidate: lowercase text, spatial log-lik of the typed word given it, log prior, kind. */
class M3Cand(val text: String, val sp: Double, val lp: Double, val kind: String)

/**
 * Port of m3dict lexicon.CandGen (GEN = qd3 7, short 2, bag_n 300): SymSpell deletes (distance <= 2, <= 3 from 7
 * letters, <= 1 up to 2 letters) + m2's bag prefilter top-300 for words of >= 6 letters, capped to the 4,000 most
 * frequent, ranked by spatial log-lik + 0.5 * log prior (ties: lexicon id); then apostrophe expansions, g-drop and the
 * top-2 space splits. Ties everywhere: (score desc, lexicon id asc).
 */
class M3CandGen(
    val L: M3Lexicon,
    private val index: SymSpellIndex,
    val sp: SpatialModel = SpatialModel(),
    val priorW: Double = 0.5,
    val maxHits: Int = 4000,
    val qd3: Int = 7,
    val bagN: Int = 300,
    val bagMin: Int = 6,
) {
    val maxd = index.maxd
    val short = index.short
    /** splits use the keyboard.py eval geometry (space bar x 3.5 .. 8.5), like the reference */
    val v2 = CandidateGeneratorV2(CandidateGenerator(L.lex, sp))

    fun deletes(s: String, d: Int): Set<String> {
        val out = HashSet<String>()
        out.add(s)
        var frontier: Set<String> = setOf(s)
        repeat(d) {
            val nxt = HashSet<String>()
            for (x in frontier) for (i in x.indices) nxt.add(x.substring(0, i) + x.substring(i + 1))
            out.addAll(nxt); frontier = nxt
        }
        return out
    }

    private fun isSubseq(q: String, w: String): Boolean {
        var j = 0
        for (c in w) { if (j < q.length && q[j] == c) j++ }
        return j == q.length
    }

    /** sorted unique lexicon ids sharing a delete with typed (lexicon.CandGen.lookup) */
    fun lookup(typed: String): IntArray {
        val s = typed.lowercase().replace("'", "")
        val d = if (s.length > short) (if (s.length >= qd3) 3 else maxd) else 1
        val got = HashSet<Int>()
        for (q in deletes(s, d)) {
            index.forKey(q) { id ->
                val w = L.stripped[id]
                val dw = if (w.length > short) maxd else 1
                val diff = w.length - q.length
                if (diff in 0..dw && isSubseq(q, w)) got.add(id)
            }
        }
        return got.toIntArray().also { it.sort() }
    }

    /** m2's cheap bag prefilter (lexicon.CandGen.bag_top): top n by (score desc, id asc), score > -1e8, sorted ids */
    fun bagTop(tl: String, n: Int): IntArray {
        val lex = L.lex
        val q = FloatArray(26)
        for (c in tl) if (c in 'a'..'z') q[c - 'a'] += 1f
        var nlq = 0f
        for (x in q) nlq += x
        val N = lex.size
        val cheap = DoubleArray(N)
        for (i in 0 until N) {
            val dn = abs(lex.nletters[i].toFloat() - nlq)
            if (dn > 2f) { cheap[i] = -1e9; continue }
            var l1 = 0f
            val o = i * 26
            for (j in 0 until 26) l1 += abs(lex.bag[o + j] - q[j])
            val c32 = -(l1 + 1.5f * dn)
            cheap[i] = c32.toDouble() + 0.15 * lex.logp[i]
        }
        val take = minOf(n, N)
        val vals = cheap.copyOf().also { it.sort() }
        val thr = vals[N - take]
        val out = ArrayList<Int>(take)
        for (i in 0 until N) if (cheap[i] > thr) out.add(i)
        for (i in 0 until N) { if (out.size >= take) break; if (cheap[i] == thr) out.add(i) }
        return out.filter { cheap[it] > -1e8 }.toIntArray().also { it.sort() }
    }

    fun spatial(tl: String, words: List<String>): DoubleArray {
        val em = sp.emitTable(KeyGeom.keyTouches(tl.lowercase()))
        return DoubleArray(words.size) { sp.score(em, sp.wordCode(words[it])!!) }
    }

    /** -> ranked candidates (typed itself excluded): top-k words, then contractions / g-drop, then <= 2 splits */
    fun cands(typed: String, k: Int = 8, splits: Boolean = true): List<M3Cand> {
        val tl = typed.lowercase()
        var ids = lookup(tl)
        val n = tl.count { it != '\'' }
        if (bagN > 0 && n >= bagMin) ids = (ids.toHashSet() + bagTop(tl, bagN).toHashSet()).toIntArray().also { it.sort() }
        if (ids.size > maxHits) ids = ids.copyOf(maxHits)
        val keep = ids.filter { L.words[it] != tl }
        val res = ArrayList<M3Cand>()
        if (keep.isNotEmpty()) {
            val words = keep.map { L.words[it] }
            val sc = spatial(tl, words)
            val rank = DoubleArray(keep.size) { sc[it] + priorW * L.logp[keep[it]] }
            val order = keep.indices.sortedWith { a, b -> if (rank[a] != rank[b]) rank[b].compareTo(rank[a]) else keep[a].compareTo(keep[b]) }
            for (j in order.take(k)) res.add(M3Cand(words[j], sc[j], L.logp[keep[j]], "word"))
        }
        val have = res.mapTo(HashSet()) { it.text }
        val extra = ArrayList<Pair<String, String>>()
        if ('\'' !in tl) L.expand[tl]?.forEach { extra.add(it to "contr") }
        if (tl.endsWith("in") && tl.length >= 4 && L.has(tl + "g")) extra.add(tl + "g" to "gdrop")
        val ex = extra.filter { it.first !in have }
        if (ex.isNotEmpty()) {
            val sc = spatial(tl, ex.map { it.first })
            ex.forEachIndexed { i, (w, kd) -> res.add(M3Cand(w, sc[i], L.logp[L.id(w)], kd)) }
        }
        if (splits && tl.isNotEmpty() && tl.all { it in 'a'..'z' } && !L.has(tl)) {
            for (c in v2.splits(KeyGeom.keyTouches(tl), tl).take(2)) {
                if (c.text in have) continue
                var lp = 0.0
                for (x in c.text.split(' ')) lp += L.logp[L.id(x)]
                res.add(M3Cand(c.text, c.sp, lp, c.kind))
            }
        }
        return res
    }
}
