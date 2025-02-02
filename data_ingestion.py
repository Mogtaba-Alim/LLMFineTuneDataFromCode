import os
import glob
import PyPDF2
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500  # Adjust as needed
OVERLAP = 50

def read_pdf(filepath: str) -> str:
    """Read a PDF file and return its text content."""
    text = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

def read_text_file(filepath: str) -> str:
    """Read a text-based file (including .md) and return its content."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_documents(folder_path: str) -> List[str]:
    """Load PDF, MD, and TXT files from a folder and return them as strings."""
    all_files = glob.glob(os.path.join(folder_path, "*.*"))
    docs = []
    for filepath in all_files:
        extension = os.path.splitext(filepath)[1].lower()
        if extension == ".pdf":
            doc_text = read_pdf(filepath)
        else:
            # Handle .txt, .md, or any text-based extension
            doc_text = read_text_file(filepath)
        if doc_text.strip():
            docs.append(doc_text)
    return docs

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple text chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_embeddings(docs: List[str], model_name: str="sentence-transformers/all-MiniLM-L6-v2"):
    """Return list of chunk embeddings and chunk texts."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=False)
    return embeddings

if __name__ == "__main__":
    data_folder = "lab_docs"  # folder with PDF/MD/TXT
    raw_docs = load_documents(data_folder)
    
    # Chunk all documents
    all_chunks = []
    for doc in raw_docs:
        all_chunks.extend(chunk_text(doc, CHUNK_SIZE, OVERLAP))
    
    # Build embeddings
    chunk_embeddings = build_embeddings(all_chunks)
    
    # Save chunks and embeddings for indexing
    np.save("chunks.npy", np.array(all_chunks, dtype=object))
    np.save("embeddings.npy", chunk_embeddings)
    print("Document loading, chunking, and embedding complete.")
