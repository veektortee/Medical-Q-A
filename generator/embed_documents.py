import os
import glob
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Settings
DOCS_PATH = "data/docs/*.txt"
CHUNK_SIZE = 250  # characters
OVERLAP = 50
INDEX_PATH = "data/faiss_index"
META_PATH = "data/metadata.pkl"

documents = []
metadata = []

# Split into overlapping chunks
def chunk_text(text, chunk_size=250, overlap=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# Read and chunk all docs
for filepath in glob.glob(DOCS_PATH):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    documents.extend(chunks)
    metadata.extend([os.path.basename(filepath)] * len(chunks))

# Embed
embeddings = model.encode(documents, show_progress_bar=True)
embeddings = normalize(embeddings)

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)

# Save metadata
with open(META_PATH, 'wb') as f:
    pickle.dump({'documents': documents, 'metadata': metadata}, f)

print(f"✅ Embedded {len(documents)} chunks and saved index + metadata.")