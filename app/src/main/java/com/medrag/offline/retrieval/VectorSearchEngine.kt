package com.medrag.offline.retrieval

import com.google.gson.Gson
import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class ScoredFact(val factId: String, val score: Float)

/**
 * Loads vectors.bin (written by step2_build_vector_store.py) fully into RAM
 * and answers top-k cosine-similarity queries. Format:
 *   [int32 numVectors little-endian][int32 dim little-endian]
 *   [numVectors * dim float32 little-endian, row-major]
 *
 * Memory budget: 8,000 facts * 384 dims * 4 bytes ≈ 12 MB - trivial even on
 * a 4GB device, which is why this is loaded eagerly rather than mmap'd.
 */
class VectorSearchEngine private constructor(
    private val matrix: FloatArray,
    private val numVectors: Int,
    private val dim: Int,
    private val factIds: List<String>,
) {

    companion object {
        fun load(vectorsBinFile: File, indexJsonFile: File): VectorSearchEngine {
            val (numVectors, dim, matrix) = readVectorsBin(vectorsBinFile)
            val indexJson = indexJsonFile.readText()
            val index = Gson().fromJson(indexJson, VectorIndexFile::class.java)
            require(index.fact_ids.size == numVectors) {
                "vectors_index.json has ${index.fact_ids.size} ids but vectors.bin has $numVectors vectors"
            }
            return VectorSearchEngine(matrix, numVectors, dim, index.fact_ids)
        }

        private fun readVectorsBin(file: File): Triple<Int, Int, FloatArray> {
            DataInputStream(FileInputStream(file)).use { stream ->
                val headerBytes = ByteArray(8)
                stream.readFully(headerBytes)
                val headerBuf = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
                val numVectors = headerBuf.int
                val dim = headerBuf.int

                val payloadBytes = ByteArray(numVectors * dim * 4)
                stream.readFully(payloadBytes)
                val payloadBuf = ByteBuffer.wrap(payloadBytes).order(ByteOrder.LITTLE_ENDIAN)
                val matrix = FloatArray(numVectors * dim)
                payloadBuf.asFloatBuffer().get(matrix)

                return Triple(numVectors, dim, matrix)
            }
        }
    }

    private data class VectorIndexFile(val fact_ids: List<String>, val dim: Int, val count: Int)

    /**
     * [queryVec] must already be L2-normalized (EmbeddingEngine guarantees this),
     * and the stored vectors were normalized in step2_build_vector_store.py, so
     * cosine similarity reduces to a plain dot product.
     */
    fun topK(queryVec: FloatArray, k: Int = 3): List<ScoredFact> {
        require(queryVec.size == dim) { "query dim ${queryVec.size} != index dim $dim" }

        // For a few thousand facts a full linear scan is well under a
        // millisecond on any phone CPU - no need for an ANN index.
        val scores = FloatArray(numVectors)
        for (i in 0 until numVectors) {
            var dot = 0f
            val base = i * dim
            for (d in 0 until dim) {
                dot += matrix[base + d] * queryVec[d]
            }
            scores[i] = dot
        }

        val topIndices = (0 until numVectors)
            .sortedByDescending { scores[it] }
            .take(k)

        return topIndices.map { ScoredFact(factIds[it], scores[it]) }
    }
}
