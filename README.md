# RAG_with_LLMs_basic
## 🧠 Retrieval-Augmented Generation (RAG) with LangChain & TinyLlama

This project demonstrates how to build an **offline RAG (Retrieval-Augmented Generation)** pipeline using:

- 💬 **Meta-llama/Llama-3.2-3B-Instruct** for local language modeling
- 🧱 **LangChain** for chaining retriever + prompt + LLM
- 📄 **PDF file ingestion and chunking**
- 🔍 **Vector search with FAISS or Chroma**
- ⚙️ Optimized to run on CPU or GPU with 4-bit quantization (`bitsandbytes`)

---

## 📂 Project Structure

<pre> project_root/ 
  ├── test.py # Entry point to run RAG loop 
  ├── src/ 
  |    ├── base/ 
  │    │    └── llm_model.py # Load Llama + quantization 
  │    └── rag/ 
  │         ├── main.py # Build RAG pipeline 
  │         ├── file_loader.py # Load & chunk PDF files 
  │         ├── vectorstore.py # FAISS/Chroma vector DB 
  │         ├── offline_rag.py # Chain prompt → LLM → output parser 
  │         └── utils.py # Text extract utilities 
  └── data_source/ 
       └── generative_ai/ # Folder containing your PDF files </pre>

## ▶️ Run the system
python test.py
