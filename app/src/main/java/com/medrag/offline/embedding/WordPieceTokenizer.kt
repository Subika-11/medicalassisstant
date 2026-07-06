package com.medrag.offline.embedding

import java.io.File

/**
 * Minimal BERT WordPiece tokenizer, sufficient to reproduce the inputs
 * sentence-transformers/all-MiniLM-L6-v2 expects (input_ids, attention_mask,
 * token_type_ids). This avoids pulling in the full HuggingFace tokenizers
 * native library just for one small vocab.
 *
 * Load vocab.txt from assets/onnx_embedder/vocab.txt (one token per line,
 * line number == token id - this is the standard BERT vocab file format).
 */
class WordPieceTokenizer(vocabFile: File, private val maxSeqLen: Int = 256) {

    private val vocab: Map<String, Int>
    private val clsId: Int
    private val sepId: Int
    private val padId: Int
    private val unkId: Int

    init {
        val lines = vocabFile.readLines()
        val map = HashMap<String, Int>(lines.size)
        for ((idx, token) in lines.withIndex()) {
            map[token] = idx
        }
        vocab = map
        clsId = vocab["[CLS]"] ?: error("vocab.txt missing [CLS] token")
        sepId = vocab["[SEP]"] ?: error("vocab.txt missing [SEP] token")
        padId = vocab["[PAD]"] ?: 0
        unkId = vocab["[UNK]"] ?: error("vocab.txt missing [UNK] token")
    }

    data class Encoding(
        val inputIds: LongArray,
        val attentionMask: LongArray,
        val tokenTypeIds: LongArray,
    )

    fun encode(text: String): Encoding {
        val basicTokens = basicTokenize(text)
        val wordpieceIds = mutableListOf(clsId)
        for (token in basicTokens) {
            wordpieceIds.addAll(wordpieceTokenize(token))
            if (wordpieceIds.size >= maxSeqLen - 1) break
        }
        wordpieceIds.add(sepId)

        val truncated = wordpieceIds.take(maxSeqLen)
        val padded = LongArray(maxSeqLen) { i ->
            if (i < truncated.size) truncated[i].toLong() else padId.toLong()
        }
        val attentionMask = LongArray(maxSeqLen) { i ->
            if (i < truncated.size) 1L else 0L
        }
        val tokenTypeIds = LongArray(maxSeqLen) { 0L }

        return Encoding(padded, attentionMask, tokenTypeIds)
    }

    /** Lowercase, split on whitespace, and pull punctuation out as its own token
     *  (matches BertTokenizer's `do_lower_case=True, basic tokenizer` behavior
     *  closely enough for short medical queries). */
    private fun basicTokenize(text: String): List<String> {
        val lowered = text.lowercase()
        val tokens = mutableListOf<String>()
        val sb = StringBuilder()
        for (ch in lowered) {
            when {
                ch.isWhitespace() -> {
                    if (sb.isNotEmpty()) { tokens.add(sb.toString()); sb.clear() }
                }
                isPunctuation(ch) -> {
                    if (sb.isNotEmpty()) { tokens.add(sb.toString()); sb.clear() }
                    tokens.add(ch.toString())
                }
                else -> sb.append(ch)
            }
        }
        if (sb.isNotEmpty()) tokens.add(sb.toString())
        return tokens
    }

    private fun isPunctuation(ch: Char): Boolean {
        val code = ch.code
        return (code in 33..47) || (code in 58..64) || (code in 91..96) || (code in 123..126)
    }

    /** Greedy longest-match-first WordPiece, with "##" continuation prefix,
     *  identical algorithm to HuggingFace's BasicWordpieceTokenizer. */
    private fun wordpieceTokenize(token: String): List<Int> {
        if (token.length > 100) return listOf(unkId)

        val output = mutableListOf<Int>()
        var start = 0
        while (start < token.length) {
            var end = token.length
            var matchedId: Int? = null
            var matchedSubstr: String? = null
            while (start < end) {
                var substr = token.substring(start, end)
                if (start > 0) substr = "##$substr"
                val id = vocab[substr]
                if (id != null) {
                    matchedId = id
                    matchedSubstr = substr
                    break
                }
                end--
            }
            if (matchedId == null) {
                return listOf(unkId)
            }
            output.add(matchedId)
            start += (matchedSubstr!!.removePrefix("##")).length
        }
        return output
    }
}
