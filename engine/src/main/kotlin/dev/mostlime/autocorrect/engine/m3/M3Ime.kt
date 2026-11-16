package dev.mostlime.autocorrect.engine.m3

import dev.mostlime.autocorrect.engine.Kind
import dev.mostlime.autocorrect.engine.SentenceFix
import dev.mostlime.autocorrect.engine.Slot
import dev.mostlime.autocorrect.engine.WordResult

object M3Tier {
    const val NONE = 0
    const val KEEP = 1
    const val SUGGEST = 2
    const val AUTO = 3
}

/**
 * m3 results in the keyboard's terms. The keyboard asks per finished word (WordResult) and, when idle, per sentence
 * (SentenceFix). The m3dict tiers map onto the existing policy: AUTO = applied in auto-correct mode, SUGGEST =
 * underlined + offered (as an unsure correction), suggest-only mode shows both as suggestions.
 */
object M3Ime {
    /** score gap of an AUTO change over the typed word: above every confidence / risky / flag threshold of the controller */
    const val AUTO_GAP = 20.0
    const val SUGGEST_GAP = 5.0

    /** text window for a word-end call: the current sentence (up to . ! ? or a newline), at most maxChars, whole words */
    fun window(text: String, maxChars: Int = 200): String {
        var start = 0
        for (i in text.indices) {
            val c = text[i]
            if (c == '\n' || ((c == '.' || c == '!' || c == '?') && i + 1 < text.length && text[i + 1].isWhitespace())) start = i + 1
        }
        var s = text.substring(start).trimStart()
        if (s.length > maxChars) {
            val cut = s.length - maxChars
            val sp = s.indexOf(' ', cut)
            s = if (sp >= 0) s.substring(sp + 1) else s.substring(cut)
        }
        return s
    }

    private fun plan(r: M3Result, j: Int) = r.plans.firstOrNull { it.j == j }

    class Alts(val texts: List<String>, val scores: DoubleArray, val chosen: Int, val tier: Int)

    /** typed word j of the window: alternatives best first, scores relative to keeping the typed word */
    fun alts(r: M3Result, j: Int, typed: String, P: M3Params): Alts {
        val p = plan(r, j)
        val auto = r.auto.firstOrNull { it.j == j }
        val sugg = r.suggest.firstOrNull { it.j == j }
        if (p == null) return Alts(listOf(typed), doubleArrayOf(0.0), 0, M3Tier.KEEP)
        // decision score of every considered alternative relative to COPY (decide's margin per alternative)
        val sc = M3Decide.altMargins(p, P)
        val texts = ArrayList<String>(); val scores = ArrayList<Double>()
        texts.add(typed); scores.add(0.0)
        val change = auto ?: sugg
        val order = p.alts.indices.filter { p.alts[it].kind != "copy" && !sc[it].isNaN() }.sortedByDescending { sc[it] }
        for (i in order) {
            val t = p.alts[i].text
            if (t == typed || t in texts) continue
            texts.add(t)
            scores.add(minOf(sc[i] - P.thSugg, -0.01))
        }
        if (change == null) return Alts(texts, scores.toDoubleArray(), 0, M3Tier.KEEP)
        var ci = texts.indexOf(change.to)
        if (ci < 0) { texts.add(change.to); scores.add(0.0); ci = texts.size - 1 }
        scores[ci] = if (auto != null) AUTO_GAP else SUGGEST_GAP
        return Alts(texts, scores.toDoubleArray(), ci, if (auto != null) M3Tier.AUTO else M3Tier.SUGGEST)
    }

    /** the decision for the last word of the window (and an AUTO change of the word before it, if any) */
    fun wordResult(r: M3Result, window: String, typed: String, P: M3Params, prevOut: String?): WordResult? {
        val words = M3Words.SEG.findAll(window).toList()
        val j = words.size - 1
        if (j < 0 || words[j].value != typed) return null
        val a = alts(r, j, typed, P)
        val slot = Slot(typed, a.texts, DoubleArray(a.texts.size), 0, a.texts.map { Kind.WORD }, null)
        val prevText = if (prevOut != null && j >= 1 && words[j - 1].value == prevOut && r.plans.any { it.j == j })
            r.auto.firstOrNull { it.j == j - 1 }?.to?.takeIf { it != prevOut } else null
        return WordResult(slot, a.scores, null, a.chosen, a.texts[a.chosen], skipped = false, cheap = false, nScored = a.texts.size,
            nCands = a.texts.size, lmMs = r.ms, recheck = null, prevText = prevText).also { it.m3Tier = a.tier }
    }

    /** every AUTO / SUGGEST change of a sentence as a SentenceFix (offsets in the typed sentence) */
    fun sentenceFixes(r: M3Result, sentence: String, P: M3Params): List<SentenceFix> {
        val words = M3Words.SEG.findAll(sentence).toList()
        val out = ArrayList<SentenceFix>()
        for (c in (r.auto + r.suggest).sortedBy { it.j }) {
            val w = words.getOrNull(c.j) ?: continue
            val a = alts(r, c.j, w.value, P)
            val fixes = a.texts.indices.sortedByDescending { a.scores[it] }.map { a.texts[it] }.filter { it.lowercase() != w.value.lowercase() }.take(3)
            if (fixes.isEmpty()) continue
            out.add(SentenceFix(w.range.first, w.value, c.to, a.scores[a.chosen], fixes))
        }
        return out
    }
}
