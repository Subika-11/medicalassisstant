package com.medrag.offline

import android.app.Application

/**
 * Kept intentionally minimal. Heavy singletons (LlamaBridge handle,
 * EmbeddingEngine, VectorSearchEngine, FactDatabase) are built lazily
 * inside MainViewModel rather than here, so a process restart after a
 * background kill doesn't re-trigger a multi-second model load before the
 * user has even opened the app.
 */
class MedicalRagApplication : Application()
