package dev.mostlime.autocorrect

import dev.mostlime.autocorrect.model.Manifest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class ManifestTest {
    private val nest = "https://autocorrect.mostlime.hackclub.app/autocorrect-android/"

    private fun bundled(): Manifest {
        val f = listOf("src/main/assets/manifest.json", "app/src/main/assets/manifest.json").map(::File).first { it.exists() }
        return Manifest.parse(f.readText())
    }

    private fun json(model: String, top: String) = """
        {"schema": 1, "default": "a", $top
         "models": [{"id": "a", "label": "A", $model
                     "graph": {"file": "a.ort", "size": 1, "sha256": "00"}, "vocab": {"file": "v.json", "size": 1, "sha256": "11"},
                     "meta": {"n_kv_heads": 2, "head_dim": 64, "conv_kernel": 7, "d_model": 384, "state_names": ["k0"]},
                     "params": {"lam": 1.0, "bias": 0.0}}]}
    """

    @Test fun perModelMirrorsComeFirst() {
        val m = Manifest.parse(json(""""base_urls": ["https://hf/a/", "https://nest/"],""", """"base_urls": ["https://nest/", "https://top2/"],"""))
        assertEquals(listOf("https://hf/a/", "https://nest/"), m.models[0].baseUrls)
        // the model's own list first, then the shared one, without repeats
        assertEquals(listOf("https://hf/a/", "https://nest/", "https://top2/"), m.mirrors(m.models[0]))
    }

    @Test fun topLevelOnlyWhenModelHasNone() {
        val m = Manifest.parse(json("", """"base_urls": ["https://nest/"],"""))
        assertTrue(m.models[0].baseUrls.isEmpty())
        assertEquals(listOf("https://nest/"), m.mirrors(m.models[0]))
        val old = Manifest.parse(json("", """"base_url": "https://old/","""))
        assertEquals(listOf("https://old/"), old.mirrors(old.models[0]))
    }

    @Test fun bundledManifestUsesHuggingFaceThenNest() {
        val m = bundled()
        val hf = mapOf("m2s_int4" to "https://huggingface.co/MostLime/keylime-m2-preview/resolve/main/",
            "block4" to "https://huggingface.co/MostLime/keylime-m1-preview/resolve/main/")
        for ((id, repo) in hf) {
            val spec = m.model(id)!!
            assertEquals(id, listOf(repo, nest), spec.baseUrls)
            assertEquals(id, repo, m.mirrors(spec).first())
        }
        assertEquals("m2s_int4", m.default)
        assertEquals("block4", m.batterySaver)
        for (spec in m.models) for (u in m.mirrors(spec)) assertFalse("$u still points at GitHub", "github.com" in u)
    }

    /** m3 lives in models_v2: shown in m1's picker slot, m2 stays the default, m1 stays for battery saver / old apps */
    @Test fun m3InModelsV2() {
        val m = bundled()
        val m3 = m.model("m3")!!
        val info = m3.m3!!
        assertEquals("block4", m3.replaces)
        assertEquals(listOf("m3.encoder.int4.onnx", "m3.decoder.int4.onnx", "vocab_m3.json", "m3_lexicon.tsv", "m3_symspell.bin", "m3_ngram.bin"),
            m3.files.map { it.name })
        assertEquals(listOf("m2s_int4", "m3"), m.picker.map { it.id })
        assertEquals("m3", m.pickFor("block4"))
        assertEquals("m2s_int4", m.pickFor("m2s_int4"))
        assertEquals("m2s_int4", m.default)
        assertEquals("block4", m.batterySaver)              // old apps: m1
        assertEquals("m3", m.pickFor(m.batterySaver))       // new apps: battery saver runs m3
        assertEquals("https://huggingface.co/MostLime/keylime-m3-preview/resolve/main/", m.mirrors(m3).first())
        assertEquals(mapOf("casual" to 256, "standard" to 257, "formal" to 258), info.styleIds)
        assertTrue(info.params.b > 0 && info.params.thSugg < info.params.thAuto)
    }

    /** what released apps (Android <= 0.1.118, Mac <= 0.2.16) read: "models" only, every entry a full step-graph entry */
    @Test fun oldAppsSeeOnlyTheirModels() {
        val f = listOf("src/main/assets/manifest.json", "app/src/main/assets/manifest.json").map(::File).first { it.exists() }
        val o = Json.parseToJsonElement(f.readText()).jsonObject
        val models = o["models"]!!.jsonArray.map { it.jsonObject }
        assertEquals(listOf("m2s_int4", "block4"), models.map { it["id"]!!.jsonPrimitive.content })
        for (e in models) for (k in listOf("graph", "vocab", "meta", "params", "label")) assertTrue("$k in ${e["id"]}", e.containsKey(k))
    }

    @Test fun unknownModelsV2TypeIsSkipped() {
        val m = Manifest.parse(json("", "\"base_urls\": [\"https://nest/\"], \"models_v2\": [{\"id\": \"z\", \"type\": \"future\", \"label\": \"Z\"}],"))
        assertEquals(listOf("a"), m.models.map { it.id })
        assertEquals(listOf("a"), m.picker.map { it.id })
    }

    @Test fun manifestComesFromNestFirst() {
        assertEquals(nest + "manifest.json", Prefs.DEFAULT_MANIFEST_URLS[0])
        assertEquals("https://github.com/Mostlime12195/keylime-models/releases/download/models-v1/manifest.json", Prefs.DEFAULT_MANIFEST_URLS[1])
    }

    /** AC_LIVE_MODELS=1: every file of every bundled model downloads from its first mirror (HF, which redirects to its CDN). */
    @Test fun liveHuggingFaceDownloads() {
        assumeTrue(System.getenv("AC_LIVE_MODELS") == "1")
        val m = bundled()
        for (spec in m.models) for (f in spec.files) {
            val u = m.mirrors(spec).first() + f.name
            val c = (URL(u).openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000; readTimeout = 30_000; setRequestProperty("Cache-Control", "no-cache")
            }
            try {
                assertEquals(u, 200, c.responseCode)
                val md = MessageDigest.getInstance("SHA-256")
                var got = 0L
                c.inputStream.use { inp ->
                    val buf = ByteArray(1 shl 16)
                    while (true) { val n = inp.read(buf); if (n < 0) break; md.update(buf, 0, n); got += n }
                }
                assertEquals(u, f.size, got)
                assertEquals(u, f.sha256, md.digest().joinToString("") { "%02x".format(it) })
            } finally {
                c.disconnect()
            }
        }
    }
}
