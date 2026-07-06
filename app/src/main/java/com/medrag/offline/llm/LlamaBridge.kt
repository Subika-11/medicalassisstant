package com.medrag.offline.llm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Thin Kotlin wrapper over the native llama.cpp JNI bridge implemented in
 * cpp/llama_jni.cpp. All heavy work happens on a single background thread
 * (Dispatchers.IO) because llama.cpp contexts are NOT thread-safe - never
 * call generate() concurrently with itself on the same handle.
 *
 * NOTE for whoever wires the native build: the exact llama.h function
 * signatures used inside llama_jni.cpp drift between llama.cpp releases.
 * If Gradle's CMake step fails with "undefined reference to llama_xxx",
 * open llama.cpp/include/llama.h at the commit you cloned and patch the
 * three or four call sites flagged with "// VERIFY AGAINST llama.h" in
 * llama_jni.cpp. The Kotlin contract below (function names + JNI types)
 * does not need to change.
 */
object LlamaBridge {

    init {
        System.loadLibrary("medrag_llama")
    }

    /**
     * JNI calls back into onToken() once per sampled token, on the same
     * background thread that called generate() (no thread-hop needed).
     * Return true to keep generating, false to stop early.
     */
    interface TokenCallback {
        fun onToken(token: String): Boolean
    }

    // --- native methods, implemented in cpp/llama_jni.cpp -----------------
    @JvmStatic external fun nativeInit(modelPath: String, nCtx: Int, nThreads: Int): Long
    @JvmStatic external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        stopToken: String,
        callback: TokenCallback,
    ): String
    @JvmStatic external fun nativeFree(handle: Long)
    @JvmStatic external fun nativePinBigCores(): Boolean
    // ------------------------------------------------------------------------

    private var handle: Long = 0L
    private var initialized = false

    suspend fun initialize(modelPath: String, nCtx: Int = 2048, nThreads: Int = 4): Boolean =
        withContext(Dispatchers.IO) {
            if (initialized) return@withContext true
            nativePinBigCores() // best-effort; safe to ignore failure on locked-down OEMs
            handle = nativeInit(modelPath, nCtx, nThreads)
            initialized = handle != 0L
            initialized
        }

    /**
     * [onToken], if provided, fires for every token as it's generated -
     * use it to push partial text into the UI so the response feels live
     * instead of appearing all at once after the full answer is ready.
     * The full, final string is still returned at the end for validation
     * (ResponseValidator needs the complete answer, not partial fragments).
     */
    suspend fun generate(
        prompt: String,
        maxTokens: Int = 256,
        temperature: Float = 0.1f,
        stopToken: String = "<|im_end|>",
        onToken: (String) -> Unit = {},
    ): String = withContext(Dispatchers.IO) {
        check(initialized) { "LlamaBridge.initialize() must succeed before generate()" }
        val callback = object : TokenCallback {
            override fun onToken(token: String): Boolean {
                onToken(token)
                return true // set to false here if you later add user-initiated cancellation
            }
        }
        nativeGenerate(handle, prompt, maxTokens, temperature, stopToken, callback)
    }

    fun shutdown() {
        if (initialized) {
            nativeFree(handle)
            handle = 0L
            initialized = false
        }
    }
}
