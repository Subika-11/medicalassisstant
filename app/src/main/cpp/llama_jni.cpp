// JNI bridge between LlamaBridge.kt and llama.cpp.
//
// Function signatures below were verified against a fresh clone of
// https://github.com/ggml-org/llama.cpp (June 2026) - see include/llama.h
// in whatever commit you vendor as the llama.cpp/ submodule. If your
// submodule pins an older release and the build fails with "no member
// named llama_model_load_from_file" or similar, the most likely culprits
// are the renames noted inline below (each has a DEPRECATED old name you
// can fall back to as a quick unblock, but prefer updating the submodule).

#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <android/log.h>

#include "llama.h"
#include "cpu_affinity.h"

#define LOG_TAG "medrag_llama_jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

    std::once_flag g_backend_init_flag;

    struct LlamaHandle {
        llama_model* model = nullptr;
        llama_context* ctx = nullptr;
        const llama_vocab* vocab = nullptr;
        int n_ctx = 0;
    };

    std::string jstring_to_utf8(JNIEnv* env, jstring jstr) {
        const char* chars = env->GetStringUTFChars(jstr, nullptr);
        std::string result(chars);
        env->ReleaseStringUTFChars(jstr, chars);
        return result;
    }

    std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text,
                                      bool add_special, bool parse_special) {
        // Standard llama.cpp idiom: a negative return value is -(required size).
        int n_tokens = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                       nullptr, 0, add_special, parse_special);
        std::vector<llama_token> tokens(n_tokens);
        int written = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                     tokens.data(), (int32_t)tokens.size(),
                                     add_special, parse_special);
        if (written < 0) {
            LOGE("llama_tokenize failed unexpectedly (returned %d)", written);
            tokens.clear();
        } else {
            tokens.resize(written);
        }
        return tokens;
    }

    std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), /*lstrip=*/0, /*special=*/true);
        if (n < 0) return "";
        return std::string(buf, n);
    }

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_medrag_offline_llm_LlamaBridge_nativeInit(
        JNIEnv* env, jclass /*clazz*/, jstring jModelPath, jint nCtx, jint nThreads) {

    std::call_once(g_backend_init_flag, []() {
        llama_backend_init();
        LOGI("llama backend initialized");
    });

    std::string model_path = jstring_to_utf8(env, jModelPath);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;       // CPU-only on a 4GB budget phone
    mparams.use_mmap = true;        // critical for low-RAM devices - avoids a full copy into the heap
    mparams.use_mlock = false;      // don't pin pages; we want the OS free to reclaim under memory pressure

    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        LOGE("failed to load model from %s", model_path.c_str());
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)nCtx;
    cparams.n_batch = (uint32_t)std::min(nCtx, 512);
    cparams.n_threads = nThreads;
    cparams.n_threads_batch = nThreads;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOGE("failed to create context");
        llama_model_free(model);
        return 0;
    }

    auto* handle = new LlamaHandle{model, ctx, llama_model_get_vocab(model), nCtx};
    LOGI("model + context ready (n_ctx=%d, n_threads=%d)", nCtx, nThreads);
    return reinterpret_cast<jlong>(handle);
}

JNIEXPORT jstring JNICALL
Java_com_medrag_offline_llm_LlamaBridge_nativeGenerate(
        JNIEnv* env, jclass /*clazz*/, jlong jHandle, jstring jPrompt,
        jint maxTokens, jfloat temperature, jstring jStopToken, jobject jCallback) {

    auto* handle = reinterpret_cast<LlamaHandle*>(jHandle);
    if (handle == nullptr || handle->ctx == nullptr) {
        return env->NewStringUTF("[error] invalid native handle");
    }

    // Look up TokenCallback.onToken(String):boolean ONCE, outside the
    // per-token loop - GetMethodID does a name/signature lookup that's
    // wasteful to repeat ~100+ times per answer. jCallback is the same
    // object instance for the whole call, so this is safe to cache locally.
    jclass callbackClass = env->GetObjectClass(jCallback);
    jmethodID onTokenMethod = env->GetMethodID(callbackClass, "onToken", "(Ljava/lang/String;)Z");
    env->DeleteLocalRef(callbackClass);
    if (onTokenMethod == nullptr) {
        LOGE("TokenCallback.onToken(String):boolean not found - JNI/Kotlin signature mismatch");
        return env->NewStringUTF("[error] callback method lookup failed");
    }

    // Fresh KV cache per call - this context is reused across multiple
    // independent user questions, and we never want stale context bleeding
    // from a previous answer into the next one.
    llama_memory_clear(llama_get_memory(handle->ctx), /*data=*/true);

    std::string prompt = jstring_to_utf8(env, jPrompt);
    std::string stop_token = jstring_to_utf8(env, jStopToken);

    // parse_special=true so the literal "<|im_start|>"/"<|im_end|>" text in
    // the prompt is mapped to Qwen's actual special tokens, not tokenized
    // as plain text. add_special=true lets the model add BOS if it expects one.
    std::vector<llama_token> tokens = tokenize(handle->vocab, prompt, /*add_special=*/true,
            /*parse_special=*/true);
    if (tokens.empty()) {
        return env->NewStringUTF("[error] tokenization failed");
    }
    if ((int)tokens.size() >= handle->n_ctx) {
        LOGE("prompt (%zu tokens) exceeds n_ctx (%d) - truncate context facts upstream",
             tokens.size(), handle->n_ctx);
        return env->NewStringUTF(
                "[error] prompt too long for context window - reduce retrieved facts or n_ctx");
    }

    // --- prefill the whole prompt in one decode call ---
    llama_batch prompt_batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(handle->ctx, prompt_batch) != 0) {
        LOGE("llama_decode failed on prompt prefill");
        return env->NewStringUTF("[error] decode failed during prompt prefill");
    }

    // --- sampler chain: temperature + repetition penalty + sampling ---
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);

    // Add repetition penalty (prevents loops)
    // last_n=64 tokens, penalty=1.1f
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));

    if (temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(/*seed=*/1234));
    }

    std::string output;
    output.reserve(maxTokens * 4);

    llama_token next_token = -1;
    for (int i = 0; i < maxTokens; ++i) {
        next_token = llama_sampler_sample(sampler, handle->ctx, -1);
        llama_sampler_accept(sampler, next_token);

        if (llama_vocab_is_eog(handle->vocab, next_token)) {
            break;
        }

        std::string piece = token_to_piece(handle->vocab, next_token);
        output += piece;

        // Stream this token back to Kotlin immediately, before the stop-token
        // check below - the caller sees partial output as it's generated
        // rather than waiting for the entire answer to finish.
        jstring jPiece = env->NewStringUTF(piece.c_str());
        jboolean shouldContinue = env->CallBooleanMethod(jCallback, onTokenMethod, jPiece);
        env->DeleteLocalRef(jPiece); // avoid exhausting the JNI local ref table over many tokens
        if (!shouldContinue) {
            LOGI("generation stopped early by callback at token %d", i);
            break;
        }

        if (!stop_token.empty() && output.size() >= stop_token.size() &&
            output.compare(output.size() - stop_token.size(), stop_token.size(), stop_token) == 0) {
            output.erase(output.size() - stop_token.size());
            break;
        }

        // feed the just-sampled token back in for the next step
        llama_batch next_batch = llama_batch_get_one(&next_token, 1);
        if (llama_decode(handle->ctx, next_batch) != 0) {
            LOGE("llama_decode failed during generation at step %d", i);
            break;
        }
    }

    llama_sampler_free(sampler);
    return env->NewStringUTF(output.c_str());
}

JNIEXPORT void JNICALL
Java_com_medrag_offline_llm_LlamaBridge_nativeFree(JNIEnv* /*env*/, jclass /*clazz*/, jlong jHandle) {
    auto* handle = reinterpret_cast<LlamaHandle*>(jHandle);
    if (handle == nullptr) return;
    if (handle->ctx) llama_free(handle->ctx);
    if (handle->model) llama_model_free(handle->model);
    delete handle;
    LOGI("native handle freed");
}

JNIEXPORT jboolean JNICALL
Java_com_medrag_offline_llm_LlamaBridge_nativePinBigCores(JNIEnv* /*env*/, jclass /*clazz*/) {
    return pin_thread_to_big_cores() ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"
