package com.medrag.offline.pipeline

import com.medrag.offline.embedding.EmbeddingEngine
import com.medrag.offline.llm.LlamaBridge
import com.medrag.offline.retrieval.FactDatabase
import com.medrag.offline.retrieval.MedicalFact
import com.medrag.offline.retrieval.VectorSearchEngine

data class RagAnswer(
    val text: String,
    val sources: List<MedicalFact>,
    val wasEmergency: Boolean,
    val grounded: Boolean = true,
)

/**
 * Top-level orchestrator matching the 6-stage architecture validated on PC
 * in step3_rag_simulation.py:
 *   1. EmergencyTriageGuard  4. PromptBuilder
 *   2. EmbeddingEngine        5. LlamaBridge (llama.cpp)
 *   3. VectorSearchEngine     6. ResponseValidator
 *
 * Construct once (e.g. in MedicalRagApplication) and reuse - all the
 * dependencies it holds are themselves long-lived, heavy objects.
 */
class RagPipeline(
    private val embeddingEngine: EmbeddingEngine,
    private val vectorSearchEngine: VectorSearchEngine,
    private val factDatabase: FactDatabase,
    private val topK: Int = 2,
) {

    companion object {
        // Cosine similarity (vectors are normalized, so this is a plain dot
        // product) below which the top retrieved fact is considered too
        // weak a match to treat as "the knowledge base actually covers
        // this" - the question falls through to general-knowledge mode
        // instead of being forced through a citation it doesn't deserve.
        // 0.45 is a starting point, not a measured constant - tune it
        // against real queries: too low lets weak matches get cited as
        // if authoritative, too high makes the app over-refuse again.
        private const val RELEVANCE_THRESHOLD = 0.45f
    }

    suspend fun answer(userQuery: String, onToken: (String) -> Unit = {}): RagAnswer {
        EmergencyTriageGuard.check(userQuery)?.let { emergencyText ->
            return RagAnswer(text = emergencyText, sources = emptyList(), wasEmergency = true)
        }

        val queryVec = embeddingEngine.embed(userQuery)
        val scored = vectorSearchEngine.topK(queryVec, k = topK)
        val isGrounded = (scored.maxOfOrNull { it.score } ?: 0f) >= RELEVANCE_THRESHOLD

        // Only pull facts into context (and only allow them to be cited)
        // when the match was actually good - a weak top-3 still shouldn't
        // get treated as evidence just because it's the best of a bad lot.
        val retrievedFacts = if (isGrounded) {
            factDatabase.getFactsByIds(scored.map { it.factId })
        } else {
            emptyList()
        }

        val prompt = PromptBuilder.build(userQuery, retrievedFacts, grounded = isGrounded)
        val rawResponse = LlamaBridge.generate(prompt, onToken = onToken)
        val validated = ResponseValidator.validate(
            rawResponse,
            retrievedFacts,
            grounded = isGrounded,
        )

        return RagAnswer(text = validated, sources = retrievedFacts, wasEmergency = false, grounded = isGrounded)
    }
}