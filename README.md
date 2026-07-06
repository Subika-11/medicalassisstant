HealthPulse:  

A privacy-preserving, 100% offline Retrieval-Augmented Generation (RAG) Android application. This app uses a localized Large Language Model (LLM) and a vector database to provide verified medical information about diabetes without requiring an internet connection.  
🚀 Key Features  
•Offline Inference: Powered by llama.cpp, running a quantized Qwen 0.8B model locally on your device's CPU.    
•Vector Search: Custom high-speed vector retrieval engine for finding relevant medical facts in milliseconds.    
•Privacy First: No data ever leaves the device. Queries are processed locally using ONNX and GGUF models.    
•Safety Guardrails:    
 ◦Emergency Triage: Instant detection of "red-flag" symptoms.    
 ◦Response Validation: Cross-references AI citations against actual retrieved facts to prevent hallucinations.  

🏗 Architecture   
The app follows a 6-stage pipeline to ensure accuracy:  
1.Emergency Triage Guard: Scans for acute symptoms and provides immediate medical disclaimers.  
2.Embedding Engine (ONNX): Converts user queries into 384-dimensional vectors using all-MiniLM-L6-v2.  
3.Vector Search: Performs a cosine-similarity scan against a local index of ~600 medical facts.  
4.Prompt Builder: Constructs a ChatML prompt containing retrieved context and system grounding instructions.  
5.LLM Inference: The LLM generates a response based only on the provided context.  
6.Response Validator: Validates that all citations (e.g., DM-000001) are authentic and not hallucinated.  

🛠 Tech Stack  
Component | Technology    
LLM Engine | llama.cpp (Android Native)    
Model Format | GGUF (Q4_K_M Quantization)    
Embedding | ONNX Runtime (Mobile)    
Database | SQLite (Metadata) + Binary Float Matrix (Vectors)      
Language | Kotlin + C++ (JNI)    

📁 Project Structure  
•app/src/main/cpp/: Native llama.cpp source and JNI bridge.  
•app/src/main/assets/: Pre-packaged models (.gguf, .onnx) and medical databases (.db, .bin).  
•app/src/main/java/com/medrag/offline/pipeline/: The core RAG pipeline logic.  
•app/src/main/java/com/medrag/offline/retrieval/: Vector search and SQLite fact retrieval.  

🚦 Performance Optimization  
The app is currently configured for optimal mobile performance:  
•Threads: nThreads = 4 for parallel CPU processing.  
•Context Window: nCtx = 1024 to reduce prefill latency.  
•Top-K: 3 facts retrieved per query to keep prompts concise.  
