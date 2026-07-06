package com.medrag.offline.embedding

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import kotlin.math.sqrt

/**
 * Loads onnx_embedder/model.onnx (exported by step2b_export_embedder_onnx.py)
 * and embeds short strings into 384-dim vectors that are directly comparable
 * (via dot product) to the vectors stored in vectors.bin, because both sides
 * use the same model + the same mean-pooling + L2-normalization recipe that
 * sentence-transformers applies internally.
 *
 * This must run on-device ONLY for queries (a handful of tokens), never for
 * bulk re-embedding - that work was already done on the PC in Step 2.
 */
class EmbeddingEngine(modelDir: File) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: WordPieceTokenizer
    val dimension: Int = 384

    init {
        val modelFile = File(modelDir, "model.onnx")
        val vocabFile = File(modelDir, "vocab.txt")
        require(modelFile.exists()) { "Missing ${modelFile.path} - run step2b_export_embedder_onnx.py" }
        require(vocabFile.exists()) { "Missing ${vocabFile.path} - run step2b_export_embedder_onnx.py" }

        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2) // embedding query is tiny, no need to hog the big cores
        }
        session = env.createSession(modelFile.absolutePath, sessionOptions)
        tokenizer = WordPieceTokenizer(vocabFile)
    }

    /** Returns an L2-normalized 384-dim embedding for [text]. */
    fun embed(text: String): FloatArray {
        val encoding = tokenizer.encode(text)
        val seqLen = encoding.inputIds.size

        val inputIdsTensor = OnnxTensor.createTensor(env, arrayOf(encoding.inputIds))
        val attentionMaskTensor = OnnxTensor.createTensor(env, arrayOf(encoding.attentionMask))
        val tokenTypeIdsTensor = OnnxTensor.createTensor(env, arrayOf(encoding.tokenTypeIds))

        inputIdsTensor.use { ids ->
            attentionMaskTensor.use { mask ->
                tokenTypeIdsTensor.use { types ->
                    val inputs = mapOf(
                        "input_ids" to ids,
                        "attention_mask" to mask,
                        "token_type_ids" to types,
                    )
                    session.run(inputs).use { results ->
                        // ORTModelForFeatureExtraction outputs "last_hidden_state"
                        // shape (1, seqLen, 384)
                        @Suppress("UNCHECKED_CAST")
                        val lastHiddenState = (results[0].value as Array<Array<FloatArray>>)[0]
                        return meanPoolAndNormalize(lastHiddenState, encoding.attentionMask, seqLen)
                    }
                }
            }
        }
    }

    private fun meanPoolAndNormalize(
        hiddenStates: Array<FloatArray>,
        attentionMask: LongArray,
        seqLen: Int,
    ): FloatArray {
        val pooled = FloatArray(dimension)
        var validTokenCount = 0f

        for (t in 0 until seqLen) {
            if (attentionMask[t] == 1L) {
                val vec = hiddenStates[t]
                for (d in 0 until dimension) {
                    pooled[d] += vec[d]
                }
                validTokenCount += 1f
            }
        }
        if (validTokenCount > 0f) {
            for (d in 0 until dimension) {
                pooled[d] /= validTokenCount
            }
        }

        var normSq = 0f
        for (d in 0 until dimension) normSq += pooled[d] * pooled[d]
        val norm = sqrt(normSq.toDouble()).toFloat().coerceAtLeast(1e-12f)
        for (d in 0 until dimension) pooled[d] /= norm

        return pooled
    }

    override fun close() {
        session.close()
    }
}
