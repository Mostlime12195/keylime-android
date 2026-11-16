package dev.mostlime.autocorrect.engine

import dev.mostlime.autocorrect.engine.m3.M3Alt
import dev.mostlime.autocorrect.engine.m3.M3Change
import dev.mostlime.autocorrect.engine.m3.M3Ime
import dev.mostlime.autocorrect.engine.m3.M3Params
import dev.mostlime.autocorrect.engine.m3.M3Plan
import dev.mostlime.autocorrect.engine.m3.M3Result
import dev.mostlime.autocorrect.engine.m3.M3Tier
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class M3ImeTest {
    private val P = M3Params(k = 3, rare = 10000.0, lowconf = -1.0, a = 0.5, b = 0.5, d = 0.0, cName = 2.0, cReal = 4.0, cRare = 8.0,
        thAuto = -5.9, thSugg = -8.1)

    @Test fun windowIsTheCurrentSentence() {
        assertEquals("and then teh", M3Ime.window("It was late. and then teh"))
        assertEquals("next line wrod", M3Ime.window("first line\nnext line wrod"))
        val long = (1..80).joinToString(" ") { "word$it" }
        val w = M3Ime.window(long, 40)
        assertTrue(w.length <= 40 && long.endsWith(w) && !w.startsWith(" "))
    }

    private fun plan(j: Int, typed: String, alts: List<M3Alt>) =
        M3Plan(j, typed, intArrayOf(0, 0), typed, nonword = true, prot = false, changed = false, rare = false, low = false, wlp = 0.0, rank = -1,
            nameLike = false, alts = alts)

    @Test fun tiersMapOntoTheKeyboardPolicy() {
        // copy S = 0; "the" S = -3 -> margin -3 >= th_auto (AUTO); "tea" S = -7 (below th_sugg relative)
        val p1 = plan(2, "teh", listOf(M3Alt("teh", "copy"), M3Alt("the", "word", dm3 = -3.0), M3Alt("tea", "word", dm3 = -7.0)))
        val p0 = plan(1, "wnet", listOf(M3Alt("wnet", "copy"), M3Alt("went", "word", dm3 = -7.0)))
        val r = M3Result("I went the", "I went teh", listOf(M3Change(2, "teh", "the", -3.0, intArrayOf(0, 0))),
            listOf(M3Change(1, "wnet", "went", -7.0, intArrayOf(0, 0))), fast = false, blocked = 0, plans = listOf(p0, p1), ms = 1.0)
        val w = M3Ime.wordResult(r, "I wnet teh", "teh", P, prevOut = "wnet")!!
        assertEquals("the", w.out)
        assertEquals(M3Tier.AUTO, w.m3Tier)
        assertTrue(w.margin >= 9.0)                     // above the controller's confident / risky gaps
        assertEquals(listOf("the", "teh", "tea"), w.ranked(3))
        assertNull(w.prevText)                          // the word before is only a SUGGEST: never edited here
        val s = M3Ime.wordResult(r, "I wnet", "wnet", P, prevOut = null)!!
        assertEquals(M3Tier.SUGGEST, s.m3Tier)
        assertEquals("went", s.out)
        val fixes = M3Ime.sentenceFixes(r, "I wnet teh", P)
        assertEquals(listOf("wnet" to "went", "teh" to "the"), fixes.map { it.from to it.to })
        assertEquals(listOf(2, 7), fixes.map { it.start })
    }

    @Test fun keepWhenNoChange() {
        val p = plan(0, "Hello", listOf(M3Alt("Hello", "copy"), M3Alt("Hell", "word", dm3 = -20.0)))
        val r = M3Result("Hello", "Hello", emptyList(), emptyList(), fast = false, blocked = 0, plans = listOf(p), ms = 1.0)
        val w = M3Ime.wordResult(r, "Hello", "Hello", P, null)!!
        assertEquals("Hello", w.out)
        assertEquals(M3Tier.KEEP, w.m3Tier)
        assertEquals(0, w.chosen)
    }
}
