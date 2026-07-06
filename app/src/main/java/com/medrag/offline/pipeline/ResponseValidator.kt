package com.medrag.offline.pipeline

/**
 * Stage 6 - the last line of defense against hallucination. Even with a
 * constrained prompt, a 0.8B model can still drift, so this checks the
 * model's own output against the fact_ids that were actually retrieved.
 */
object ResponseValidator {

    private val FACT_ID_PATTERN = Regex("""DM-\d{6}""")

    // Any specific numeric clinical figure (dose, mg/mcg/IU amount, mg/dL or
    // mmol/L reading, etc.) is the highest-risk category of hallucination -
    // a wrong number is more dangerous than a wrong qualitative statement.
    // This is checked in BOTH modes, but only ever fires in general-knowledge
    // mode in practice since grounded answers cite real source numbers.
    private val RISKY_NUMERIC_PATTERN = Regex(
        """\b\d+(\.\d+)?\s?(mg|mcg|ml|units?|iu|mmol/l|mg/dl|%)\b""",
        RegexOption.IGNORE_CASE,
    )

    fun validate(rawResponse: String, allowedFactIds: List<String>, grounded: Boolean): String {
        val trimmed = rawResponse.trim()

        if (trimmed.contains(PromptBuilder.REFUSAL_TEXT)) {
            return trimmed // explicit, well-formed refusal is always valid
        }

        return if (grounded) {
            validateGrounded(trimmed, allowedFactIds)
        } else {
            validateGeneralKnowledge(trimmed)
        }
    }

    private fun validateGrounded(trimmed: String, allowedFactIds: List<String>): String {
        val citedIds = allowedFactIds.filter { trimmed.contains(it) }
        if (citedIds.isEmpty()) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: no retrieved fact_id was cited)"
        }

        val allDmIdsInText = FACT_ID_PATTERN.findAll(trimmed).map { it.value }.toSet()
        val unauthorized = allDmIdsInText - allowedFactIds.toSet()
        if (unauthorized.isNotEmpty()) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: cited unretrieved id(s) $unauthorized)"
        }

        return trimmed
    }

    private fun validateGeneralKnowledge(trimmed: String): String {
        if (!trimmed.startsWith(PromptBuilder.GENERAL_KNOWLEDGE_TAG)) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: missing general-knowledge disclaimer tag)"
        }

        if (RISKY_NUMERIC_PATTERN.containsMatchIn(trimmed)) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: ungrounded answer stated a specific " +
                    "clinical number - rejected rather than risk an unverified figure)"
        }

        return trimmed
    }
}