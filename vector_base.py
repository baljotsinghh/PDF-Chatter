import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)  # L2 distance metric

    def add_embeddings(self, embeddings: np.ndarray):
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        D, I = self.index.search(query_embedding, top_k)
        return I, D  # indices and distances