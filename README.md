# 🩺 MediQuery — Medical Q&A System with RAG and OpenAI

MediQuery is an intelligent medical question-answering system that leverages Retrieval-Augmented Generation (RAG) to provide helpful, accurate, and grounded responses to health-related queries. It combines the power of OpenAI's LLM with a custom-embedded knowledge base built from the MedQuAD dataset (a curated set of 47,000+ Q&A pairs from trusted medical sources).

---

## 🔍 How It Works

- **Dataset:** Uses the MedQuAD dataset containing thousands of real medical Q&A pairs.
- **Embedding:** All questions and answers are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Store:** Embedded data is stored in a FAISS index for fast similarity search.
- **RAG Pipeline:** On each query:
  1. The system retrieves the most relevant documents from the vector store.
  2. These are passed to an OpenAI model (e.g., `gpt-4o`) using a custom prompt template.
  3. The model generates an answer based only on the retrieved context.
- **Disclaimer:** The assistant always includes a reminder that this is not medical advice.

---

## 🚀 Features

- 🔎 Context-aware answers
- ⚡ Fast FAISS-powered retrieval
- 🧠 Custom prompt template for safe generation
- 📁 Custom-trained on embedded MedQuAD data
- 🌐 Runs locally: CLI & Streamlit web app

---

## 📁 Project Structure

```
project/
├── retriever/
│   ├── medical_qa.py   # Main RAG pipeline using LangChain, FAISS, OpenAI
│   ├── answer.py        # Simple CLI: test questions locally
├── generator/
│   └── extract_medquad.py # Extract MedQuAD Q&A and generate embeddings
├── app/
│   └── app.py          # Streamlit frontend for easy user interaction
├── data/
│   ├── docs/medquad.txt  # Raw MedQuAD Q&A
│   ├── faiss_index       # FAISS index file
│   └── metadata.pkl      # Stored chunks & metadata

```

---

## 🖥️ How to Run

1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run Streamlit App**
    ```bash
    streamlit run app.py
    ```
    ➡️ Visit `http://localhost:8501` in your browser, type a question, get an answer!

3. Run CLI test
   ```bash
   python retriever/answer.py
   ```
   ➡️ Try simple console-based queries to validate retrieval & generation

4. Regenerate embeddings (if needed)
   ```bash
   python generator/embed_documents.py
   ```
   ➡️ Extract Q&A pairs and rebuild the FAISS index.

6. **Ask health-related questions**  
   The model will respond based on real medical literature and retrieved context.

---

## ⚙️ Tech Stack
- ✅ LangChain — RAG orchestration

- ✅ SentenceTransformers — Embeddings

- ✅ FAISS — Vector similarity search

- ✅ OpenAI API — LLM completions

- ✅ Streamlit — User-friendly UI


## 🛑 Disclaimer

MediQuery is not a substitute for professional medical advice. All answers are generated from medical literature and are intended for informational purposes only. Always consult a qualified healthcare provider for any medical concerns.
