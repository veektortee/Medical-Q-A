import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from openai import OpenAI
import os

client = OpenAI()

# Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("data/faiss_index")
with open("data/metadata.pkl", "rb") as f:
    meta = pickle.load(f)

documents = meta["documents"]
metadata = meta["metadata"]

# Get user question
query = input("Ask a medical question: ")
query_vec = model.encode([query])
query_vec = normalize(query_vec)

# Retrieve top-k
k = 4
D, I = index.search(query_vec, k)
context = "\n\n".join([documents[i] for i in I[0]])

# Prompt
system_prompt = "You are a helpful, non-alarming medical assistant. Answer questions based only on the provided medical text. Always include a short disclaimer at the end that this is not medical advice."
user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

response = client.chat.completions.create(model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
],
temperature=0.3)

print("\n💬 Answer:\n")
print(response.choices[0].message.content)
