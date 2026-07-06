package com.medrag.offline.pipeline

import com.medrag.offline.retrieval.MedicalFact

/**
 * Stage 4. Builds the exact same ChatML-formatted, context-only prompt that
 * was validated in step3_rag_simulation.py. Keep SYSTEM_PROMPT text
 * identical on both sides if you tune it - it directly affects validation
 * in ResponseValidator (the refusal sentence must match verbatim).
 */
object PromptBuilder {

    const val REFUSAL_TEXT =
        "I don't have enough verified information to answer that. " +
                "Please consult a healthcare professional."

    // Required at the start of any answer given in general-knowledge (ungrounded)
    // mode, so the UI/validator/user can always tell sourced facts apart from
    // the model's own unverified training knowledge.
    const val GENERAL_KNOWLEDGE_TAG = "[GENERAL KNOWLEDGE - NOT FROM VERIFIED SOURCES]"

    private const val SYSTEM_PROMPT_GROUNDED = (
            "You are a helpful offline diabetes assistant. Answer the user's " +
                    "question using ONLY the provided CONTEXT. First, provide a clear, " +
                    "helpful response based on the facts. Then, at the very end of your " +
                    "response, list the fact_id(s) you used. Every claim must be traceable " +
                    "to a fact_id. If the CONTEXT doesn't have the answer, say: " +
                    "'$REFUSAL_TEXT' and nothing else. Never use outside knowledge. " +
                    "Never give dosages. Format sources at the end as: Sources: DM-000001."
            )

    // Used only when retrieval found nothing relevant enough to cite (see
    // RagPipeline's relevance threshold). The model may answer from its own
    // training knowledge, but under tighter guardrails than the grounded
    // path: no specific numbers for anything clinical, and the answer must
    // visibly mark itself as unverified rather than reading like a sourced fact.
    private const val SYSTEM_PROMPT_GENERAL = (
            "You are an offline diabetes information assistant. No verified source " +
                    "document was found for this question, so you must answer from your " +
                    "own general medical knowledge instead - but you must start your reply " +
                    "with exactly the tag '$GENERAL_KNOWLEDGE_TAG' on its own line. Stay " +
                    "strictly within general diabetes/health education. Use qualified, " +
                    "non-definitive language (\"typically\", \"in general\", \"often\") " +
                    "rather than stating things as certain. NEVER state a specific number " +
                    "for a medication dose, blood glucose threshold, lab value, or any " +
                    "other clinical figure - describe these only in qualitative terms and " +
                    "say a clinician must confirm the actual number. If you are not " +
                    "reasonably confident in an answer even at this general level, reply " +
                    "with exactly: '$REFUSAL_TEXT' instead. End every answer with a " +
                    "reminder to confirm with a healthcare professional."
            )

    fun build(userQuery: String, contextFacts: List<MedicalFact>, grounded: Boolean): String {
        val systemPrompt: String
        val userBlock: String

        if (grounded) {
            val contextBlock = if (contextFacts.isEmpty()) {
                "(no relevant facts found)"
            } else {
                contextFacts.joinToString("\n") { fact ->
                    "[${fact.factId}] ${fact.factText} (Source: ${fact.sourceOrganization})"
                }
            }
            systemPrompt = SYSTEM_PROMPT_GROUNDED
            userBlock = "CONTEXT:\n$contextBlock\n\nQUESTION: $userQuery"
        } else {
            systemPrompt = SYSTEM_PROMPT_GENERAL
            userBlock = "QUESTION: $userQuery"
        }

        return buildString {
            append("<|im_start|>system\n").append(systemPrompt).append("<|im_end|>\n")
            append("<|im_start|>user\n").append(userBlock).append("<|im_end|>\n")
            append("<|im_start|>assistant\n")
        }
    }
}