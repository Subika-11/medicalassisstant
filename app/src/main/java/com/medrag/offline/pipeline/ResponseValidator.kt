package com.medrag.offline.pipeline

import com.medrag.offline.retrieval.MedicalFact

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

    fun validate(rawResponse: String, allowedFacts: List<MedicalFact>, grounded: Boolean): String {
        val trimmed = rawResponse.trim()

        if (trimmed.contains(PromptBuilder.REFUSAL_TEXT)) {
            return trimmed // explicit, well-formed refusal is always valid
        }

        return if (grounded) {
            validateGrounded(trimmed, allowedFacts)
        } else {
            validateGeneralKnowledge(trimmed)
        }
    }

    private fun validateGrounded(trimmed: String, allowedFacts: List<MedicalFact>): String {
        val allowedIds = allowedFacts.map { it.factId }
        val citedById = allowedIds.filter { trimmed.contains(it) }

        // Fallback for 0.8B model: Match full name OR acronyms (like CDC, ADA, NIDDK, WHO)
        val citedByOrg = allowedFacts.filter { fact ->
            val org = fact.sourceOrganization ?: return@filter false
            // 1. Direct match (e.g. "CDC" or "NIDDK")
            if (trimmed.contains(org, ignoreCase = true)) return@filter true

            // 2. Acronym match: if org is "Centers for Disease Control (CDC)", check for "CDC"
            val acronymMatch = Regex("""\(([^)]+)\)""").find(org)
            val acronym = acronymMatch?.groupValues?.get(1)
            acronym != null && acronym.length >= 2 && trimmed.contains(acronym, ignoreCase = true)
        }

        if (citedById.isEmpty() && citedByOrg.isEmpty()) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: no retrieved fact_id or organization was cited)"
        }

        val allDmIdsInText = FACT_ID_PATTERN.findAll(trimmed).map { it.value }.toSet()
        val unauthorized = allDmIdsInText - allowedIds.toSet()
        if (unauthorized.isNotEmpty()) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: cited unretrieved id(s) $unauthorized)"
        }

        return trimmed
    }

    private fun validateGeneralKnowledge(trimmed: String): String {
        // If the model forgot the tag, but the answer is safe (checked below),
        // we'll just prepend it ourselves instead of refusing a helpful answer.
        val hasTag = trimmed.startsWith(PromptBuilder.GENERAL_KNOWLEDGE_TAG)

        if (RISKY_NUMERIC_PATTERN.containsMatchIn(trimmed)) {
            return "${PromptBuilder.REFUSAL_TEXT} (validator: ungrounded answer stated a specific " +
                    "clinical number - rejected rather than risk an unverified figure)"
        }

        // Auto-fix: if tag is missing, add it. Otherwise return as-is.
        return if (hasTag) trimmed else "${PromptBuilder.GENERAL_KNOWLEDGE_TAG}\n\n$trimmed"
    }
}