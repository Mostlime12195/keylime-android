package dev.mostlime.autocorrect.engine.m3

import java.math.BigDecimal
import java.math.RoundingMode

/** m3dict decode.py: word rules, the per-message constraint context and the piece-feeding state machine. */
object M3Words {
    val SEG = Regex("[A-Za-z]+(?:'[A-Za-z]+)*")
    val WS_TOK = Regex("\\S+")
    val PROT = Regex("[/@_#:\\\\]|\\.[A-Za-z]|[A-Za-z][0-9]|[0-9][A-Za-z]|^&|www|http")
    val I_WORDS = setOf("i", "i'm", "i'll", "i'd", "i've")
    /** apostrophe-less contractions: legal output words (casual keeps them), not in the shipping lexicon */
    val CASUAL_OK: Set<String> = ("dont cant wont im ive youre theyre isnt doesnt didnt wasnt werent arent hasnt havent hadnt " +
        "shouldnt couldnt wouldnt thats whats theres heres wheres whos youve youll youd theyve theyll theyd weve itll thatll " +
        "mustnt hows whys whens shes hes").split(' ').toSet()

    fun isLetter(c: Char) = c in 'a'..'z' || c in 'A'..'Z'

    /** Python round(x, n) (exact decimal value of the double, half-even) */
    fun round(x: Double, n: Int): Double = if (x.isNaN() || x.isInfinite()) x else BigDecimal(x).setScale(n, RoundingMode.HALF_EVEN).toDouble()

    fun applyCase(word: String, pattern: String): String {
        if (word in I_WORDS || (word.split('\'')[0] == "i" && '\'' in word)) return "I" + word.substring(1)
        val letters = pattern.filter { it.isLetter() }
        if (letters.length > 1 && letters.all { it.isUpperCase() }) return word.uppercase()
        if (letters.isNotEmpty() && letters[0].isUpperCase()) return word.take(1).uppercase() + word.drop(1)
        return word
    }

    /** Python str.isupper() on ASCII words: some letter, no lowercase letter */
    fun isUpper(s: String) = s.any { it.isLetter() } && s.none { it.isLowerCase() }

    /** per SEG word of typed text: does its whitespace token look like a URL / handle / code / number */
    fun protectedSlots(text: String): List<Boolean> {
        val out = ArrayList<Boolean>()
        for (m in WS_TOK.findAll(text)) {
            val p = PROT.containsMatchIn(m.value)
            repeat(SEG.findAll(m.value).count()) { out.add(p) }
        }
        return out
    }

    fun variant(out: String, span: IntArray, rep: String) = out.substring(0, span[0]) + rep + out.substring(span[1])
}

/** Per-message constraint context (decode.Ctx): typed copies, user words, injected candidates, the split rule. */
class M3Ctx(private val L: M3Lexicon, typedText: String, user: Collection<String> = emptyList(), injected: Collection<String> = emptyList()) {
    val words = HashSet<String>()
    private val prefixes = HashSet<String>()
    val cap: Int
    private val splitWords: List<String>

    init {
        val tw = M3Words.SEG.findAll(typedText).map { it.value.lowercase() }.toCollection(LinkedHashSet())
        val ul = user.map { it.lowercase() }.toSet()
        words.addAll(tw); words.addAll(ul)
        for (c in injected) for (m in M3Words.SEG.findAll(c)) words.add(m.value.lowercase())
        words.addAll(M3Words.CASUAL_OK)
        for (w in words) for (k in 1..w.length) prefixes.add(w.substring(0, k))
        cap = maxOf(4, tw.maxOfOrNull { w -> w.count { it != '\'' } } ?: 4) + 4
        splitWords = tw.filter { it.length >= 3 && (L.has(it) || it in ul) }
    }

    fun validWord(w: String): Boolean {
        if (w in words) return true
        if (L.has(w)) return w.count { it != '\'' } <= cap
        for (s in POSS) {
            if (w.endsWith(s) && w.length > s.length) {
                val b = w.substring(0, w.length - s.length)
                if (L.has(b) || b in words) return true
            }
        }
        return false
    }

    fun validPrefix(p: String): Boolean {
        if (p in prefixes) return true
        if (L.shortestCompletion(p) <= cap) return true
        val i = p.lastIndexOf('\'')
        if (i > 0) {
            val r = p.substring(i + 1)
            if (r == "" || r == "s") return validWord(p.substring(0, i))
        }
        return false
    }

    fun rems(w: String): List<String>? {
        val r = splitWords.filter { it.length > w.length && it.startsWith(w) }.map { it.substring(w.length) }
        return r.ifEmpty { null }
    }

    companion object { val POSS = listOf("'s", "'") }
}

/** decode.feed state: open word (lowercase, may end in '), split remainders when it started, remainders after the last word, only spaces since. */
data class M3State(val cur: String, val srem: List<String>?, val rem: List<String>?, val gap: Boolean) {
    companion object { val START = M3State("", null, null, false) }
}

class FeedResult(val ok: Boolean, val split: Boolean, val st: M3State)

object M3Feed {
    fun feed(ctx: M3Ctx, st: M3State, s: String): FeedResult {
        var cur = st.cur
        var srem = st.srem
        var rem = st.rem
        var gap = st.gap
        for (ch in s) {
            if (M3Words.isLetter(ch)) {
                if (cur.isEmpty()) srem = if (gap) rem else null
                cur += ch.lowercaseChar()
            } else if (ch == '\'' && cur.isNotEmpty() && cur.last() != '\'') {
                cur += "'"
            } else {
                if (cur.isNotEmpty()) {
                    val w = cur.trimEnd('\'')
                    if (!ctx.validWord(w)) return FeedResult(false, false, st)
                    rem = ctx.rems(w); gap = true; cur = ""; srem = null
                    if (s.isNotEmpty() && ch != ' ') gap = false
                    if (ch == '\'') gap = false
                } else if (ch != ' ') {
                    gap = false
                }
            }
        }
        var split = false
        if (cur.isNotEmpty()) {
            if (!ctx.validPrefix(cur)) return FeedResult(false, false, st)
            val sr = srem
            if (sr != null) {
                val c = cur.replace("'", "")
                split = sr.any { it.startsWith(c) }
            }
        }
        return FeedResult(true, split, M3State(cur, srem, rem, gap))
    }

    fun finishOk(ctx: M3Ctx, st: M3State) = st.cur.isEmpty() || ctx.validWord(st.cur.trimEnd('\''))
}

/** The scorer's Needleman-Wunsch with word indices (decode.align_idx). */
object M3Align {
    fun lev(a: String, b: String): Int {
        if (a.isEmpty()) return b.length
        if (b.isEmpty()) return a.length
        var prev = IntArray(b.length + 1) { it }
        var cur = IntArray(b.length + 1)
        for (i in 1..a.length) {
            cur[0] = i
            for (j in 1..b.length) {
                val c = if (a[i - 1] == b[j - 1]) 0 else 1
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
            }
            val t = prev; prev = cur; cur = t
        }
        return prev[b.length]
    }

    /** rapidfuzz Levenshtein.normalized_distance */
    fun ned(a: String, b: String): Double {
        val m = maxOf(a.length, b.length)
        return if (m == 0) 0.0 else lev(a, b).toDouble() / m
    }

    fun alignIdx(aw: List<String>, bw: List<String>, GAP: Double = 0.6, MULTI: Double = 0.45, SUBW: Double = 0.6): List<List<Int>> {
        val a = aw.map { it.lowercase() }
        val b = bw.map { it.lowercase() }
        val n = a.size
        val m = b.size
        val INF = 1e9
        val D = Array(n + 1) { DoubleArray(m + 1) { INF } }
        // back pointer: op (1 del, 2 ins, 3 sub, 4 merge, 5 split), di, dj
        val bop = Array(n + 1) { IntArray(m + 1) }
        val bdi = Array(n + 1) { IntArray(m + 1) }
        val bdj = Array(n + 1) { IntArray(m + 1) }
        D[0][0] = 0.0
        for (i in 0..n) for (j in 0..m) {
            if (i == 0 && j == 0) continue
            var best = INF
            var op = 0; var di = 0; var dj = 0
            if (i > 0 && D[i - 1][j] + GAP < best) { best = D[i - 1][j] + GAP; op = 1; di = 1; dj = 0 }
            if (j > 0 && D[i][j - 1] + GAP < best) { best = D[i][j - 1] + GAP; op = 2; di = 0; dj = 1 }
            if (i > 0 && j > 0) {
                var c = D[i - 1][j - 1] + SUBW * ned(a[i - 1], b[j - 1])
                if (c < best) { best = c; op = 3; di = 1; dj = 1 }
                for (k in 2..3) {
                    if (i >= k) {
                        c = D[i - k][j - 1] + SUBW * ned(a.subList(i - k, i).joinToString(""), b[j - 1]) + MULTI * (k - 1)
                        if (c < best) { best = c; op = 4; di = k; dj = 1 }
                    }
                    if (j >= k) {
                        c = D[i - 1][j - k] + SUBW * ned(a[i - 1], b.subList(j - k, j).joinToString("")) + MULTI * (k - 1)
                        if (c < best) { best = c; op = 5; di = 1; dj = k }
                    }
                }
            }
            D[i][j] = best; bop[i][j] = op; bdi[i][j] = di; bdj[i][j] = dj
        }
        val ops = ArrayList<IntArray>()
        var i = n
        var j = m
        while (i > 0 || j > 0) {
            val op = bop[i][j]; val di = bdi[i][j]; val dj = bdj[i][j]
            ops.add(intArrayOf(op, i - di, i, j - dj, j)); i -= di; j -= dj
        }
        ops.reverse()
        val out = List(n) { ArrayList<Int>() }
        for (o in ops) {
            if (o[0] == 1 || o[0] == 2) continue
            for (x in o[3] until o[4]) out[o[1]].add(x)
        }
        return out
    }
}

/** An alternative for an uncertain slot: [text, kind, dm3, spatial, log prior, ngram delta] (features rounded to 4 dp). */
class M3Alt(val text: String, val kind: String, var dm3: Double = 0.0, var sp: Double = 0.0, var pr: Double = 0.0, var ng: Double = 0.0)

class M3Plan(
    val j: Int, val typed: String, val span: IntArray, val outWord: String, val nonword: Boolean, val prot: Boolean,
    val changed: Boolean, val rare: Boolean, val low: Boolean, val wlp: Double, val rank: Int, val nameLike: Boolean,
    var alts: List<M3Alt>,
)

class M3Params(
    val k: Int, val rare: Double, val lowconf: Double, val a: Double, val b: Double, val d: Double,
    val cName: Double, val cReal: Double, val cRare: Double, val thAuto: Double, val thSugg: Double,
)

class M3Decision(val text: String, val margin: Double, val kind: String, val index: Int)

/** tune.py: the decision rule (which alternatives count, scores, margin vs COPY) and applying the AUTO tier. */
object M3Decide {
    fun candsOn(p: M3Plan, P: M3Params) = p.nonword || (p.rare && (p.rank >= P.rare || p.wlp < P.lowconf))

    fun decide(p: M3Plan, P: M3Params): M3Decision {
        var best: M3Alt? = null
        var bs = -1e18
        var bi = 0
        var scCopy = 0.0
        var ncand = 0
        for ((i, x) in p.alts.withIndex()) {
            if (x.kind != "copy" && x.kind != "greedy") {
                if (!candsOn(p, P)) continue
                ncand++
                if (ncand > P.k) continue
            }
            val s = x.dm3 + P.a * x.sp + P.b * x.ng + P.d * x.pr
            if (x.kind == "copy") {
                scCopy = s + (if (p.nameLike) P.cName else 0.0) + (if (p.nonword) 0.0 else if (p.changed) P.cReal else P.cRare)
            } else if (s > bs) { best = x; bs = s; bi = i }
        }
        val b = best ?: return M3Decision(p.alts[0].text, -1e18, "copy", 0)
        return M3Decision(b.text, bs - scCopy, b.kind, bi)
    }

    /** per alternative: S(alt) - S(COPY) as decide() scores them (NaN for the copy and for alternatives decide skips) */
    fun altMargins(p: M3Plan, P: M3Params): DoubleArray {
        val keep = decideAlts(p, P).toSet()
        val s = DoubleArray(p.alts.size) { Double.NaN }
        var scCopy = 0.0
        for ((i, x) in p.alts.withIndex()) {
            if (i !in keep) continue
            val v = x.dm3 + P.a * x.sp + P.b * x.ng + P.d * x.pr
            if (x.kind == "copy") scCopy = v + (if (p.nameLike) P.cName else 0.0) + (if (p.nonword) 0.0 else if (p.changed) P.cReal else P.cRare)
            else s[i] = v
        }
        for (i in s.indices) if (!s[i].isNaN()) s[i] -= scCopy
        return s
    }

    /** indices decide() looks at (tune.decide_alts) */
    fun decideAlts(p: M3Plan, P: M3Params): List<Int> {
        val out = ArrayList<Int>()
        var ncand = 0
        for ((i, x) in p.alts.withIndex()) {
            if (x.kind != "copy" && x.kind != "greedy") {
                if (!candsOn(p, P)) continue
                ncand++
                if (ncand > P.k) continue
            }
            out.add(i)
        }
        return out
    }

    /** greedy text with every change of margin >= th applied, the others reverted to the typed word */
    fun apply(o0: String, plans: List<M3Plan>, P: M3Params, th: Double): String {
        val edits = ArrayList<Pair<IntArray, String>>()
        for (p in plans) {
            val d = decide(p, P)
            val tgt = if (d.kind != "copy" && d.margin >= th) d.text else p.alts[0].text
            if (tgt != p.outWord) edits.add(p.span to tgt)
        }
        var o = o0
        for ((s, t) in edits.sortedByDescending { it.first[0] }) o = o.substring(0, s[0]) + t + o.substring(s[1])
        return o
    }

    /**
     * casual: drop generator alternatives that only add or move apostrophes (im -> i'm); a typed word that is itself a
     * casual contraction (CASUAL_OK: dont, im, ...) is no correction site at all (its slot is dropped)
     */
    fun casualFilter(plans: List<M3Plan>): List<M3Plan> = plans.filter { p ->
        '\'' in p.typed || p.typed.lowercase() !in M3Words.CASUAL_OK
    }.map { p ->
        val t = p.typed.replace("'", "").lowercase()
        M3Plan(p.j, p.typed, p.span, p.outWord, p.nonword, p.prot, p.changed, p.rare, p.low, p.wlp, p.rank, p.nameLike,
            p.alts.filter { it.kind == "copy" || it.kind == "greedy" || it.text.replace("'", "").lowercase() != t })
    }
}
