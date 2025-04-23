import streamlit as st
from pdf_handler import PDFTextExtractor
from chunker import TextChunker
from embedder import EmbeddingManager
from vector_base import VectorStore
import torch
import os
from gemini_wrapper import GeminiAPI

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
def main():
    st.set_page_config(page_title="PDF Text Chunker", layout="centered")
    st.title("📄 PDF Text Extractor & Chunker")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        extractor = PDFTextExtractor(uploaded_file)
        raw_text = extractor.extract_text()

        if raw_text.strip() == "":
            st.warning("No text found in PDF.")
            return

        st.subheader("Raw Text Preview (first 500 chars)")
        st.text(raw_text[:500] + "...")

        chunker = TextChunker(raw_text, chunk_size=500, overlap=50)
        chunks = chunker.chunk_text()

        st.subheader(f"📦 Chunked Output ({len(chunks)} chunks)")
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1}"):
                st.write(chunk)

        st.success("Text successfully extracted and chunked!")
        
        # After chunking:
        st.subheader("🔐 Generating Embeddings...")
        embedder = EmbeddingManager()
        embeddings = embedder.encode_chunks(chunks)

        st.success(f"✅ Created {embeddings.shape[0]} embeddings")

        # Store in FAISS
        st.subheader("📚 Indexing in Vector Store...")
        vs = VectorStore(dim=embeddings.shape[1])
        vs.add_embeddings(embeddings)
        st.success("💾 Stored in FAISS vector DB")

        # Optional query
        with st.expander("🔍 Try Similarity Search"):
            user_query = st.text_input("Enter a query:")
            if user_query:
                query_emb = embedder.encode_chunks([user_query])
                indices, distances = vs.search(query_emb, top_k=3)
                
                st.write("Most similar chunks:")
                for i in indices[0]:
                    st.markdown(f"> {chunks[i]}")
                    
                for i in indices[0]:
                    st.markdown(f"> {chunks[i]}")

                if st.button("💬 Pass to LLM"):
                    gemini = GeminiAPI()
                    try:
                        response = gemini.ask([chunks[i] for i in indices[0]], user_query)
                        st.subheader("🧠 LLM's Response")
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error from Gemini API: {e}")

if __name__ == "__main__":
    main()