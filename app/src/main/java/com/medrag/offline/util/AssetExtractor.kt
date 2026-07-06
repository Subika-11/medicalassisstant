package com.medrag.offline.util

import android.content.Context
import java.io.File

/**
 * Android cannot mmap or random-access files living inside the APK's
 * assets/ archive efficiently (and SQLite/llama.cpp need a real filesystem
 * path, not an InputStream). So on first launch we copy each large asset
 * once into context.filesDir and reuse it on every subsequent launch.
 *
 * IMPORTANT: the corresponding file extensions (.gguf, .db, .bin, .onnx)
 * MUST be marked noCompress in app/build.gradle, or Android's aapt will
 * gzip them inside the APK, which both bloats peak RAM during extraction
 * and can silently truncate files over the old 1GB AAPT asset limit on
 * some Android Gradle Plugin versions. See README.md "Gradle gotchas".
 */
object AssetExtractor {

    /**
     * Copies assets/[assetRelativePath] to filesDir/[assetRelativePath] if
     * not already present (matched by file size as a cheap "already done"
     * check - good enough since these are versioned build assets, not
     * user-mutable files).
     */
    fun extractIfNeeded(context: Context, assetRelativePath: String): File {
        val destFile = File(context.filesDir, assetRelativePath)
        destFile.parentFile?.mkdirs()

        val assetManager = context.assets
        val assetSize = assetManager.openFd(assetRelativePath).use { it.length }

        if (destFile.exists() && destFile.length() == assetSize) {
            return destFile
        }

        assetManager.open(assetRelativePath).use { input ->
            destFile.outputStream().use { output ->
                input.copyTo(output, bufferSize = 1 shl 20) // 1MB buffer for big model files
            }
        }
        return destFile
    }

    /** Extracts every file under an assets/ subfolder (e.g. "onnx_embedder/"). */
    fun extractDirIfNeeded(context: Context, assetDirRelativePath: String): File {
        val assetManager = context.assets
        val fileNames = assetManager.list(assetDirRelativePath) ?: emptyArray()
        var lastDest: File? = null
        for (fileName in fileNames) {
            lastDest = extractIfNeeded(context, "$assetDirRelativePath/$fileName")
        }
        return lastDest?.parentFile ?: File(context.filesDir, assetDirRelativePath)
    }
}
