import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load extracted medical data
with open("medical_data.json", "r") as file:
    medical_data = json.load(file)

# Initialize Sentence Transformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert medical text into vectors
documents = [entry["content"] for entry in medical_data]
document_vectors = embedder.encode(documents)

# Store vectors in FAISS
dimension = document_vectors.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(document_vectors))

# Save FAISS index
faiss.write_index(faiss_index, "medical_faiss.index")

print("✅ Medical data indexed in FAISS.")
