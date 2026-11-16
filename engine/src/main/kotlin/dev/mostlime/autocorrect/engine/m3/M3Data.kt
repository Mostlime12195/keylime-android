package dev.mostlime.autocorrect.engine.m3

import dev.mostlime.autocorrect.engine.Lexicon
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.nio.ShortBuffer

/**
 * The m3 shipping lexicon (m3_lexicon.tsv: 100k lowercase words + counts, count-sorted) with what the constrained
 * decode needs on top of the shared [Lexicon] (words, log priors, letter bags, spatial codes):
 *  - shortest completion (letters, apostrophes not counted) of every prefix: sorted word list + a min segment tree
 *    (m3dict lexicon.Dict.prefixes, no 233k-entry map)
 *  - apostrophe expansions (dont -> don't)
 */
class M3Lexicon(val lex: Lexicon) {
    val words get() = lex.words
    val logp get() = lex.logp
    val size get() = lex.size
    val minLogp: Double = lex.logp.min()
    /** words with the apostrophes removed */
    val stripped: Array<String> = Array(lex.size) { lex.words[it].replace("'", "") }

    private val sorted: Array<String>
    private val tree: IntArray
    private val n2: Int

    /** stripped form -> lexicon words with apostrophes, lexicon order */
    val expand: Map<String, List<String>>

    init {
        val order = (0 until lex.size).sortedWith { a, b -> lex.words[a].compareTo(lex.words[b]) }
        sorted = Array(order.size) { lex.words[order[it]] }
        var n = 1
        while (n < order.size) n = n shl 1
        n2 = n
        tree = IntArray(2 * n) { NONE }
        for (i in order.indices) tree[n + i] = lex.nletters[order[i]]
        for (i in n - 1 downTo 1) tree[i] = minOf(tree[2 * i], tree[2 * i + 1])
        val e = HashMap<String, MutableList<String>>()
        for (w in lex.words) if ('\'' in w) e.getOrPut(w.replace("'", "")) { ArrayList(1) }.add(w)
        expand = e
    }

    fun has(w: String) = lex.index.containsKey(w)
    fun id(w: String): Int = lex.index[w] ?: -1

    /** Shortest completion length of prefix p among lexicon words (NONE = no word starts with p). */
    fun shortestCompletion(p: String): Int {
        val lo = lowerBound(p)
        val hi = lowerBound(p + '\u007f')
        if (lo >= hi) return NONE
        var l = lo + n2
        var r = hi - 1 + n2
        var m = NONE
        while (l <= r) {
            if (l and 1 == 1) { m = minOf(m, tree[l]); l++ }
            if (r and 1 == 0) { m = minOf(m, tree[r]); r-- }
            l = l shr 1; r = r shr 1
        }
        return m
    }

    private fun lowerBound(p: String): Int {
        var lo = 0
        var hi = sorted.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (sorted[mid] < p) lo = mid + 1 else hi = mid
        }
        return lo
    }

    companion object {
        /** Python's dict.get(p, 99) default */
        const val NONE = 99
    }
}

internal fun le(buf: ByteBuffer): ByteBuffer = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN)

/** little-endian view starting at byte offset `at` (Buffer casts: no covariant JDK 9+ overloads, older Android lacks them) */
internal fun at(buf: ByteBuffer, at: Int): ByteBuffer {
    val d = le(buf)
    (d as java.nio.Buffer).position(at)
    return d.slice().order(ByteOrder.LITTLE_ENDIAN)
}

internal fun fnv64(s: String): Long {
    var h = -0x340d631b7bdddcdbL   // 0xcbf29ce484222325
    for (b in s.toByteArray(Charsets.UTF_8)) {
        h = h xor (b.toLong() and 0xff)
        h *= 0x100000001b3L
    }
    return h
}

/**
 * m3_symspell.bin: the delete index of lexicon.CandGen (maxd 2; 1 for words of <= `short` letters). Keys are the high
 * 32 bits of FNV-1a 64 of each delete string; a hit is verified (the query delete must be a subsequence of the word,
 * at most d letters shorter), so 32-bit collisions never add candidates.
 */
class SymSpellIndex(buf0: ByteBuffer) {
    val nWords: Int
    val maxd: Int
    val short: Int
    private val keys: IntBuffer
    private val ids: IntBuffer
    private val n: Int

    init {
        val b = le(buf0)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "KLSS") { "not a KLSS file" }
        require(b.int == 1) { "KLSS version" }
        nWords = b.int; n = b.int; maxd = b.int; short = b.int
        val base = b.position()
        keys = at(b, base).asIntBuffer()
        ids = at(b, base + 4 * n).asIntBuffer()
    }

    /** word ids whose deletes share delete string q (unverified candidates of key == hash32(q)) */
    fun forKey(q: String, out: (Int) -> Unit) {
        val k = (fnv64(q) ushr 32).toInt()
        var lo = 0
        var hi = n
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (Integer.compareUnsigned(keys.get(mid), k) < 0) lo = mid + 1 else hi = mid
        }
        var i = lo
        while (i < n && keys.get(i) == k) { out(ids.get(i)); i++ }
    }
}

/**
 * m3_ngram.bin (m3dict_cpt/qkn.py): pruned KN word trigram in backoff form, every value a multiple of 1/1024, so the
 * sums are exact and identical in every port.
 */
class QuantTrigram(buf0: ByteBuffer) {
    val V: Int
    private val ids: HashMap<String, Int>
    private val p1: ShortBuffer
    private val g2: ShortBuffer
    private val boff: IntBuffer
    private val bent: IntBuffer
    private val huoff: IntBuffer
    private val hent: IntBuffer
    private val toff: IntBuffer
    private val tent: IntBuffer

    init {
        val b = le(buf0)
        val magic = ByteArray(4).also { b.get(it) }
        require(String(magic, Charsets.US_ASCII) == "KLQN") { "not a KLQN file" }
        require(b.int == 1) { "KLQN version" }
        V = b.int
        val nb = b.int; val nh = b.int; val nt = b.int; val nv = b.int
        val vb = ByteArray(nv).also { b.get(it) }
        val words = String(vb, Charsets.UTF_8).split('\n')
        require(words.size == V)
        ids = HashMap(V * 2)
        for (i in words.indices) ids[words[i]] = i
        var o = b.position() + ((4 - nv % 4) % 4)
        fun sh(cnt: Int): ShortBuffer { val s = at(b, o).asShortBuffer(); o += 2 * cnt; o += (4 - (2 * cnt) % 4) % 4; return s }
        fun ints(cnt: Int): IntBuffer { val s = at(b, o).asIntBuffer(); o += 4 * cnt; return s }
        p1 = sh(V); g2 = sh(V)
        boff = ints(V + 1); bent = ints(nb)
        huoff = ints(V + 1); hent = ints(nh)
        toff = ints(nh + 1); tent = ints(nt)
    }

    fun wid(w: String): Int {
        if (w.isNotEmpty() && w[0].isDigit()) return 2
        return ids[w.lowercase()] ?: 0
    }

    private fun q15(e: Int) = -(e and 0x7FFF).toDouble() / SCALE
    private fun q16(s: Short) = -(s.toInt() and 0xFFFF).toDouble() / SCALE

    /** entry index in [lo, hi) whose high 17 bits == x (entries sorted by them), else -1 */
    private fun find(buf: IntBuffer, lo: Int, hi: Int, x: Int): Int {
        var a = lo
        var b = hi
        while (a < b) {
            val m = (a + b) ushr 1
            if ((buf.get(m) ushr 15) < x) a = m + 1 else b = m
        }
        return if (a < hi && (buf.get(a) ushr 15) == x) a else -1
    }

    fun lp(u: Int, v: Int, w: Int): Double {
        val h = find(hent, huoff.get(u), huoff.get(u + 1), v)
        var back = 0.0
        if (h >= 0) {
            val t = find(tent, toff.get(h), toff.get(h + 1), w)
            if (t >= 0) return q15(tent.get(t))
            back = q15(hent.get(h))
        }
        val bi = find(bent, boff.get(v), boff.get(v + 1), w)
        val b2 = if (bi >= 0) q15(bent.get(bi)) else q16(g2.get(v)) + q16(p1.get(w))
        return back + b2
    }

    /** sentence log-prob of lowercase words: history <s> <s>, OOV (id 0) costs LP_OOV */
    fun sentLp(words: List<String>): Double {
        var u = 1
        var v = 1
        var s = 0.0
        for (w0 in words) {
            val w = wid(w0)
            s += if (w == 0) LP_OOV else lp(u, v, w)
            u = v; v = w
        }
        return s
    }

    companion object {
        const val SCALE = 1024.0
        const val LP_OOV = -16.0
    }
}
