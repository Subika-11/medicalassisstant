"""
STEP 3 - Constrained RAG Prompt & Inference Pipeline (PC simulation)
=======================================================================
Simulates exactly what the Android app will do at runtime, so you can
validate retrieval quality and prompt behavior before touching Kotlin.

Pipeline stages (mirrors the Android architecture 1:1):
    1. EmergencyTriageGuard - hard-coded refusal for red-flag symptoms
    2. embed_query()         - MiniLM embedding of the user's question
    3. retrieve_top_k()      - cosine similarity search over vectors.bin
    4. build_prompt()        - strict Qwen ChatML prompt, context-only
    5. llama-cpp generate()  - run Qwen3.5-0.8B GGUF
    6. ResponseValidator     - reject answers that cite no retrieved fact_id

Run:
    pip install llama-cpp-python sentence-transformers numpy
    # place qwen_3.5_0.8b_q4_k_m.gguf in this folder, then:
    python step3_rag_simulation.py
"""

import json
import re
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"
DB_PATH = OUTPUT_DIR / "medical_facts.db"
VECTORS_PATH = OUTPUT_DIR / "vectors.bin"
INDEX_PATH = OUTPUT_DIR / "vectors_index.json"
GGUF_PATH = Path(__file__).parent / "qwen_3.5_0.8b_q4_k_m.gguf"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
N_CTX = 2048
N_THREADS = 4

# ---------------------------------------------------------------------
# STAGE 1 - Emergency triage guard
# ---------------------------------------------------------------------
# Anything that looks like an acute emergency must NEVER be answered by
# retrieval+generation. It gets a hard-coded, deterministic response that
# tells the user to seek immediate in-person/emergency care instead.

EMERGENCY_KEYWORDS = [
    "unconscious", "passed out", "seizure", "can't breathe", "cannot breathe",
    "chest pain", "severe confusion", "won't wake up", "blood sugar over 400",
    "blood sugar above 400", "ketones", "fruity breath", "diabetic coma",
    "severe vomiting", "suicidal", "overdose",
]

EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please call your "
    "local emergency number or go to the nearest emergency room right away. "
    "This app cannot assess emergencies and must not be used as a "
    "substitute for emergency medical care."
)


def emergency_triage_guard(user_query: str) -> str | None:
    lowered = user_query.lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in lowered:
            return EMERGENCY_RESPONSE
    return None


# ---------------------------------------------------------------------
# STAGE 2 - Query embedding
# ---------------------------------------------------------------------

_embed_model = None


def embed_query(text: str) -> np.ndarray:
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    vec = _embed_model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32)


# ---------------------------------------------------------------------
# STAGE 3 - Vector retrieval (cosine similarity == dot product, vectors
# are pre-normalized in Step 2)
# ---------------------------------------------------------------------

def load_vector_store():
    with open(VECTORS_PATH, "rb") as f:
        n, dim = struct.unpack("<ii", f.read(8))
        matrix = np.frombuffer(f.read(), dtype=np.float32).reshape(n, dim)
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index = json.load(f)
    return matrix, index["fact_ids"]


def retrieve_top_k(query_vec: np.ndarray, matrix: np.ndarray, fact_ids: list[str],
                   k: int = TOP_K) -> list[tuple[str, float]]:
    scores = matrix @ query_vec  # (N,) cosine similarities
    top_indices = np.argsort(-scores)[:k]
    return [(fact_ids[i], float(scores[i])) for i in top_indices]


def fetch_fact_rows(fact_ids: list[str]) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = []
    for fid in fact_ids:
        cur.execute("SELECT * FROM medical_facts WHERE fact_id = ?", (fid,))
        row = cur.fetchone()
        if row:
            rows.append(dict(row))
    conn.close()
    return rows


# ---------------------------------------------------------------------
# STAGE 4 - Constrained prompt builder (Qwen ChatML template)
# ---------------------------------------------------------------------

# Cosine similarity (vectors are pre-normalized, so this is a plain dot
# product) below which the top retrieved fact is too weak a match to treat
# as "the knowledge base covers this" - falls through to general-knowledge
# mode instead of being forced through a citation it doesn't deserve.
# Starting point, not a measured constant - tune against real queries.
RELEVANCE_THRESHOLD = 0.45

GENERAL_KNOWLEDGE_TAG = "[GENERAL KNOWLEDGE - NOT FROM VERIFIED SOURCES]"

SYSTEM_PROMPT_GROUNDED = (
    "You are an offline diabetes information assistant. You may ONLY use "
    "the numbered CONTEXT facts below to answer. Every claim in your "
    "answer must be traceable to a fact_id in the CONTEXT. If the CONTEXT "
    "does not contain enough information to answer the question, you must "
    "reply with exactly: 'I don't have enough verified information to "
    "answer that. Please consult a healthcare professional.' Never use "
    "outside knowledge. Never give specific medication dosages. Always end "
    "your answer with a line listing the fact_id(s) you used, formatted as "
    "Sources: DM-000001, DM-000002."
)

# Used only when retrieval found nothing relevant enough to cite. The model
# may answer from its own training knowledge, but under tighter guardrails:
# no specific numbers for anything clinical, and the answer must visibly
# mark itself as unverified rather than reading like a sourced fact.
SYSTEM_PROMPT_GENERAL = (
    "You are an offline diabetes information assistant. No verified source "
    "document was found for this question, so you must answer from your "
    f"own general medical knowledge instead - but you must start your reply "
    f"with exactly the tag '{GENERAL_KNOWLEDGE_TAG}' on its own line. Stay "
    "strictly within general diabetes/health education. Use qualified, "
    "non-definitive language (\"typically\", \"in general\", \"often\") "
    "rather than stating things as certain. NEVER state a specific number "
    "for a medication dose, blood glucose threshold, lab value, or any "
    "other clinical figure - describe these only in qualitative terms and "
    "say a clinician must confirm the actual number. If you are not "
    "reasonably confident in an answer even at this general level, reply "
    "with exactly: 'I don't have enough verified information to answer "
    "that. Please consult a healthcare professional.' instead. End every "
    "answer with a reminder to confirm with a healthcare professional."
)


def build_prompt(user_query: str, context_facts: list[dict], grounded: bool) -> str:
    if grounded:
        if not context_facts:
            context_block = "(no relevant facts found)"
        else:
            context_block = "\n".join(
                f"[{f['fact_id']}] {f['fact_text']} (Source: {f['source_organization']})"
                for f in context_facts
            )
        system_prompt = SYSTEM_PROMPT_GROUNDED
        user_block = f"CONTEXT:\n{context_block}\n\nQUESTION: {user_query}"
    else:
        system_prompt = SYSTEM_PROMPT_GENERAL
        user_block = f"QUESTION: {user_query}"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


# ---------------------------------------------------------------------
# STAGE 5 - Generation via llama-cpp-python
# ---------------------------------------------------------------------

def generate(prompt: str) -> str:
    from llama_cpp import Llama
    if not GGUF_PATH.exists():
        print(
            f"[ERROR] {GGUF_PATH} not found. Place your Qwen GGUF file next "
            f"to this script before running generation.",
            file=sys.stderr,
        )
        sys.exit(1)

    llm = Llama(
        model_path=str(GGUF_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    result = llm(
        prompt,
        max_tokens=256,
        temperature=0.1,
        stop=["<|im_end|>"],
    )
    return result["choices"][0]["text"].strip()


# ---------------------------------------------------------------------
# STAGE 6 - Response validator
# ---------------------------------------------------------------------

REFUSAL_TEXT = (
    "I don't have enough verified information to answer that. "
    "Please consult a healthcare professional."
)

# Any specific numeric clinical figure (dose, mg/mcg/IU amount, mg/dL or
# mmol/L reading, etc.) is the highest-risk hallucination category - a wrong
# number is more dangerous than a wrong qualitative statement. Checked only
# in general-knowledge mode in practice, since grounded answers cite real
# source numbers.
RISKY_NUMERIC_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|ml|units?|iu|mmol/l|mg/dl|%)\b", re.IGNORECASE
)


def validate_response(response_text: str, allowed_fact_ids: list[str], grounded: bool) -> str:
    """Grounded mode: reject if no retrieved fact_id is cited, or an
    unretrieved id is cited (likely hallucinated). General-knowledge mode:
    reject if the disclaimer tag is missing, or a specific clinical number
    is stated without a verified source behind it."""
    if REFUSAL_TEXT.strip() in response_text:
        return response_text  # explicit refusal is always valid

    if grounded:
        cited_ids = [tok for tok in allowed_fact_ids if tok in response_text]
        if not cited_ids:
            return REFUSAL_TEXT + " (validator: no retrieved fact_id was cited)"

        all_dm_ids_in_text = set(re.findall(r"DM-\d{6}", response_text))
        unauthorized = all_dm_ids_in_text - set(allowed_fact_ids)
        if unauthorized:
            return REFUSAL_TEXT + f" (validator: cited unretrieved id(s) {unauthorized})"

        return response_text
    else:
        if not response_text.strip().startswith(GENERAL_KNOWLEDGE_TAG):
            return REFUSAL_TEXT + " (validator: missing general-knowledge disclaimer tag)"

        if RISKY_NUMERIC_PATTERN.search(response_text):
            return REFUSAL_TEXT + (
                " (validator: ungrounded answer stated a specific clinical "
                "number - rejected rather than risk an unverified figure)"
            )

        return response_text


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def answer_query(user_query: str, matrix: np.ndarray, fact_ids: list[str]) -> str:
    triage = emergency_triage_guard(user_query)
    if triage:
        return triage

    query_vec = embed_query(user_query)
    top_results = retrieve_top_k(query_vec, matrix, fact_ids, k=TOP_K)
    top_score = top_results[0][1] if top_results else 0.0
    is_grounded = top_score >= RELEVANCE_THRESHOLD

    retrieved_ids = [fid for fid, _ in top_results] if is_grounded else []
    retrieved_facts = fetch_fact_rows(retrieved_ids) if is_grounded else []

    print(f"\n[Retrieved facts] grounded={is_grounded} (top_score={top_score:.3f}, "
          f"threshold={RELEVANCE_THRESHOLD})")
    for fid, score in top_results:
        print(f"  {fid}  (score={score:.3f})")

    prompt = build_prompt(user_query, retrieved_facts, grounded=is_grounded)
    raw_response = generate(prompt)
    final_response = validate_response(raw_response, retrieved_ids, grounded=is_grounded)
    return final_response


def main():
    if not VECTORS_PATH.exists() or not DB_PATH.exists():
        print(
            "[ERROR] Run step1_scrape_facts.py and step2_build_vector_store.py "
            "first to produce medical_facts.db / vectors.bin.",
            file=sys.stderr,
        )
        sys.exit(1)

    matrix, fact_ids = load_vector_store()
    print(f"[INFO] loaded vector store with {len(fact_ids)} facts")

    test_queries = [
        "What are the risk factors for Type 2 diabetes?",
        "What is the difference between type 1 and type 2 diabetes?",
    ]
    for q in test_queries:
        print(f"\n{'=' * 70}\nQUESTION: {q}")
        answer = answer_query(q, matrix, fact_ids)
        print(f"\nANSWER:\n{answer}")


if __name__ == "__main__":
    main()
