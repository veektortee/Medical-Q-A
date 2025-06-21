import os
import pickle
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Load FAISS index and metadata
with open("data/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

index = faiss.read_index("data/faiss_index")

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Wrap documents
docs = [Document(page_content=chunk) for chunk in meta["documents"]]
index_to_docstore_id = {i: str(i) for i in range(len(docs))}
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

# Wrap FAISS index
db = FAISS(
    embedding_model,
    index,
    docstore,
    index_to_docstore_id
)

# LLM
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0.3, openai_api_key=api_key)

custom_prompt = PromptTemplate.from_template(
    """You are a helpful, non-alarming medical assistant. 
    Answer questions based only on the provided context. Say "I'm sorry, I can't help you with that request. Please consult a qualified healthcare provider if you need help." if the answer is not found in the context.
    Always include "⚠️ Disclaimer: I'm not a medical professional. If you need help, please consult a qualified healthcare provider." only after retrieving an answer. 
    Don't include a disclaimer if you don't find an answer in the context.

    Context:
    {context}

    Question: {question}
    Answer:
    """
)

# Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

def generate_answer(question: str) -> str:
    """Generate an answer to a medical question."""
    result = qa_chain.invoke({"query": question})
    return result["result"]

"""# Ask
question = input("Ask a medical question: ")
result = qa_chain.invoke({"query": question})

# Output
print("\n💬 Answer:\n")
print(result["result"])
"""
