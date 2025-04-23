from typing import List

class TextChunker:
    def __init__(self, text: str, chunk_size: int = 500, overlap: int = 50):
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self) -> List[str]:
        words = self.text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = words[start:end]
            chunks.append(" ".join(chunk))
            start += self.chunk_size - self.overlap  # move with overlap

        return chunks