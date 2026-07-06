package com.medrag.offline.retrieval

import android.database.sqlite.SQLiteDatabase
import java.io.File

data class MedicalFact(
    val factId: String,
    val factText: String,
    val category: String?,
    val sourceOrganization: String?,
    val sourceTitle: String?,
    val sourceUrl: String?,
    val retrievedDate: String?,
    val version: String?,
)

/**
 * Thin read-only wrapper around medical_facts.db (already populated by
 * step2_build_vector_store.py - this class never writes to the DB).
 */
class FactDatabase private constructor(private val db: SQLiteDatabase) : AutoCloseable {

    companion object {
        fun open(dbFile: File): FactDatabase {
            require(dbFile.exists()) { "medical_facts.db not found at ${dbFile.path}" }
            val db = SQLiteDatabase.openDatabase(
                dbFile.absolutePath,
                null,
                SQLiteDatabase.OPEN_READONLY,
            )
            return FactDatabase(db)
        }
    }

    fun getFactsByIds(factIds: List<String>): List<MedicalFact> {
        if (factIds.isEmpty()) return emptyList()
        val placeholders = factIds.joinToString(",") { "?" }
        val cursor = db.rawQuery(
            "SELECT fact_id, fact_text, category, source_organization, source_title, " +
                "source_url, retrieved_date, version FROM medical_facts " +
                "WHERE fact_id IN ($placeholders)",
            factIds.toTypedArray(),
        )

        // Preserve the rank order the caller passed in (SQL's IN clause does not).
        val byId = HashMap<String, MedicalFact>()
        cursor.use {
            while (it.moveToNext()) {
                val fact = MedicalFact(
                    factId = it.getString(0),
                    factText = it.getString(1),
                    category = it.getString(2),
                    sourceOrganization = it.getString(3),
                    sourceTitle = it.getString(4),
                    sourceUrl = it.getString(5),
                    retrievedDate = it.getString(6),
                    version = it.getString(7),
                )
                byId[fact.factId] = fact
            }
        }
        return factIds.mapNotNull { byId[it] }
    }

    override fun close() {
        db.close()
    }
}
