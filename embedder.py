from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_chunks(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings)