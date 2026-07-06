package com.medrag.offline

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.medrag.offline.embedding.EmbeddingEngine
import com.medrag.offline.llm.LlamaBridge
import com.medrag.offline.pipeline.RagPipeline
import com.medrag.offline.retrieval.FactDatabase
import com.medrag.offline.retrieval.VectorSearchEngine
import com.medrag.offline.util.AssetExtractor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed class UiState {
    object LoadingModel : UiState()
    object Ready : UiState()
    data class Answering(val question: String, val partialText: String = "") : UiState()
    data class Answered(val question: String, val answerText: String, val sourceCount: Int) : UiState()
    data class Error(val message: String) : UiState()
}

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val _uiState = MutableStateFlow<UiState>(UiState.LoadingModel)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    private var ragPipeline: RagPipeline? = null
    private var embeddingEngine: EmbeddingEngine? = null
    private var factDatabase: FactDatabase? = null

    init {
        bootstrap()
    }

    private fun bootstrap() {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val context = getApplication<Application>()

                    // 1) Pull the heavy prepackaged assets out of the APK once.
                    val modelFile = AssetExtractor.extractIfNeeded(context, "qwen3.5-0.8b-instruct-Q4_K_M.gguf")
                    val dbFile = AssetExtractor.extractIfNeeded(context, "medical_facts.db")
                    val vectorsFile = AssetExtractor.extractIfNeeded(context, "vectors.bin")
                    val indexFile = AssetExtractor.extractIfNeeded(context, "vectors_index.json")
                    val onnxDir = AssetExtractor.extractDirIfNeeded(context, "onnx_embedder")

                    // 2) Bring up retrieval-side components (fast, no model loading delay).
                    val db = FactDatabase.open(dbFile)
                    val vectorSearch = VectorSearchEngine.load(vectorsFile, indexFile)
                    val embedder = EmbeddingEngine(onnxDir)

                    factDatabase = db
                    embeddingEngine = embedder

                    // 3) Bring up the LLM last - this is the slow part (model load).
                    val ok = LlamaBridge.initialize(modelFile.absolutePath, nCtx = 1024, nThreads = 4)
                    if (!ok) {
                        _uiState.value = UiState.Error("Failed to load the language model.")
                        return@withContext
                    }

                    ragPipeline = RagPipeline(embedder, vectorSearch, db)
                }
                _uiState.value = UiState.Ready
            } catch (t: Throwable) {
                _uiState.value = UiState.Error(t.message ?: "Unknown startup error")
            }
        }
    }

    fun ask(question: String) {
        val pipeline = ragPipeline ?: return
        if (question.isBlank()) return

        viewModelScope.launch {
            _uiState.value = UiState.Answering(question)
            val partial = StringBuilder()
            try {
                // onToken fires on the IO dispatcher thread inside pipeline.answer();
                // MutableStateFlow.value is safe to set from any thread.
                // Note: this shows the model's raw, unvalidated output as it streams -
                // if ResponseValidator later rejects the full answer (e.g. no cited
                // fact_id), the UI will visibly swap to the refusal text once
                // UiState.Answered lands. That's an accepted tradeoff for making the
                // common case feel responsive.
                val result = pipeline.answer(question) { token ->
                    partial.append(token)
                    _uiState.value = UiState.Answering(question, partial.toString())
                }
                _uiState.value = UiState.Answered(
                    question = question,
                    answerText = result.text,
                    sourceCount = result.sources.size,
                )
            } catch (t: Throwable) {
                _uiState.value = UiState.Error(t.message ?: "Unknown error while answering")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        LlamaBridge.shutdown()
        embeddingEngine?.close()
        factDatabase?.close()
    }
}
