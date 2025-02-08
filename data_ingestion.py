#!/usr/bin/env python3
"""
data_ingestion.py

This script loads documents (PDF, .txt, .md), extracts text using PyMuPDF, and splits text into overlapping chunks
using LangChain's RecursiveCharacterTextSplitter (if available) for state‐of‐the‐art text splitting.
Then it builds dense embeddings using SentenceTransformer ("all-mpnet-base-v2").
"""

import os
import glob
import logging
import numpy as np
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from tqdm.auto import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Parameters
CHUNK_SIZE = 500       # desired maximum chunk size (characters)
CHUNK_OVERLAP = 50     # overlap between chunks (characters)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DATA_FOLDER = "lab_docs"   # folder containing PDF/MD/TXT files

# Try to import LangChain's text splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    logging.warning("LangChain not installed. Please install with 'pip install langchain'. Falling back to a simple splitter.")
    RecursiveCharacterTextSplitter = None

def read_pdf(filepath: str) -> str:
    """Read a PDF file using PyMuPDF and return its text content."""
    try:
        doc = fitz.open(filepath)
        text = []
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        logging.error(f"Error reading PDF {filepath}: {e}")
        return ""

def read_text_file(filepath: str) -> str:
    """Read a text-based file (.txt, .md, etc.) and return its content."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        return ""

def load_documents(folder_path: str) -> List[str]:
    """Load PDF, MD, and TXT files from a folder and return a list of document texts."""
    folder = Path(folder_path)
    docs = []
    for filepath in folder.glob("*.*"):
        extension = filepath.suffix.lower()
        if extension == ".pdf":
            doc_text = read_pdf(str(filepath))
        else:
            doc_text = read_text_file(str(filepath))
        if doc_text.strip():
            docs.append(doc_text)
    return docs

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into chunks.
    If LangChain's RecursiveCharacterTextSplitter is available, use it;
    otherwise, fall back to a simple character-based splitter.
    """
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
    else:
        # Simple fallback splitter
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
    return chunks

def build_embeddings(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """Compute embeddings for a list of texts using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype="float32")

if __name__ == "__main__":
    logging.info("Loading documents from %s", DATA_FOLDER)
    raw_docs = load_documents(DATA_FOLDER)

    all_chunks = []
    logging.info("Splitting documents into chunks...")
    for doc in tqdm(raw_docs, desc="Documents"):
        chunks = split_text(doc, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    logging.info("Total chunks created: %d", len(all_chunks))
    logging.info("Building embeddings for chunks...")
    chunk_embeddings = build_embeddings(all_chunks)

    # Save chunks and embeddings for later use
    np.save("chunks.npy", np.array(all_chunks, dtype=object))
    np.save("embeddings.npy", chunk_embeddings)
    logging.info("Document ingestion complete: chunks and embeddings saved.")
