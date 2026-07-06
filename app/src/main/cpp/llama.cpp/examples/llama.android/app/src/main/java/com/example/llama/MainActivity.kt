package com.medrag.offline

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private val viewModel: MainViewModel by viewModels()

    private lateinit var textStatus: TextView
    private lateinit var textAnswer: TextView
    private lateinit var editQuestion: EditText
    private lateinit var buttonAsk: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        textStatus = findViewById(R.id.textStatus)
        textAnswer = findViewById(R.id.textAnswer)
        editQuestion = findViewById(R.id.editQuestion)
        buttonAsk = findViewById(R.id.buttonAsk)

        buttonAsk.setOnClickListener {
            val question = editQuestion.text.toString()
            viewModel.ask(question)
        }

        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.STARTED) {
                viewModel.uiState.collect { state -> render(state) }
            }
        }
    }

    private fun render(state: UiState) {
        when (state) {
            is UiState.LoadingModel -> {
                textStatus.text = getString(R.string.status_loading_model)
                setInputEnabled(false)
            }
            is UiState.Ready -> {
                textStatus.text = getString(R.string.status_ready)
                setInputEnabled(true)
            }
            is UiState.Answering -> {
                textStatus.text = "Thinking about: ${state.question}"
                textAnswer.text = state.partialText
                setInputEnabled(false)
            }
            is UiState.Answered -> {
                textStatus.text = getString(R.string.status_ready)
                textAnswer.text = state.answerText
                setInputEnabled(true)
                editQuestion.text.clear()
            }
            is UiState.Error -> {
                textStatus.text = "Error: ${state.message}"
                setInputEnabled(false)
            }
        }
    }

    private fun setInputEnabled(enabled: Boolean) {
        editQuestion.isEnabled = enabled
        buttonAsk.isEnabled = enabled
    }
}
