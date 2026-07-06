package com.medrag.offline.pipeline

/**
 * Stage 1 of the pipeline. Runs BEFORE any retrieval or generation.
 * Matches the same keyword list validated in step3_rag_simulation.py -
 * keep these two lists in sync if you edit one.
 */
object EmergencyTriageGuard {

    private val EMERGENCY_KEYWORDS = listOf(
        "unconscious", "passed out", "seizure", "can't breathe", "cannot breathe",
        "chest pain", "severe confusion", "won't wake up", "blood sugar over 400",
        "blood sugar above 400", "ketones", "fruity breath", "diabetic coma",
        "severe vomiting", "suicidal", "overdose",
    )

    const val EMERGENCY_RESPONSE = (
        "This sounds like it could be a medical emergency. Please call your " +
        "local emergency number or go to the nearest emergency room right away. " +
        "This app cannot assess emergencies and must not be used as a " +
        "substitute for emergency medical care."
    )

    /** Returns the emergency response if [userQuery] matches a red flag, else null. */
    fun check(userQuery: String): String? {
        val lowered = userQuery.lowercase()
        return if (EMERGENCY_KEYWORDS.any { lowered.contains(it) }) EMERGENCY_RESPONSE else null
    }
}
