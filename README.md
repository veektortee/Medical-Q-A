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
- 🧠 Backed by GPT-4o
- 📁 Custom-trained on embedded MedQuAD data
- 🌐 Deployed via Streamlit for easy demo and interaction

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

3. **Ask health-related questions**  
   The model will respond based on real medical literature and retrieved context.

---

## 📁 Project Structure

```
project/
├── backend/
│   └── medical_qa.py       # Core RAG logic and model invocation
├── data/
│   ├── metadata.pkl        # Embedded MedQuAD metadata
│   └── faiss_index         # FAISS vector store
└── streamlit_app/
    └── app.py              # Frontend UI via Streamlit
```

---

## 🛑 Disclaimer

This system is **not a substitute for professional medical advice**. Answers are based on context retrieved from medical literature and are intended
