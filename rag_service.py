#!/usr/bin/env python3
"""
rag_service.py

This script implements a state-of-the-art retrieval-augmented generation (RAG) service.
It loads a prebuilt FAISS HNSW index and document chunks, retrieves the most relevant context
using a high-quality SentenceTransformer, builds an enriched prompt, and generates an answer
using a fine-tuned LLM loaded with Hugging Face Transformers.
"""

import faiss
import numpy as np
import torch
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Configuration parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "MogtabaAlim/llama3.18B-BHK-Lab-Data-Fine-tunedByMogtaba" 
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
EMBEDDING_DIM = 768

def load_faiss_index(index_file: str = INDEX_FILE):
    """Load the FAISS index from disk."""
    try:
        index = faiss.read_index(index_file)
        logging.info("FAISS index loaded from %s.", index_file)
        return index
    except Exception as e:
        logging.error("Error loading FAISS index: %s", e)
        raise

def load_chunks(chunks_file: str = CHUNKS_FILE):
    """Load the document chunks from disk."""
    try:
        chunks = np.load(chunks_file, allow_pickle=True)
        logging.info("Chunks loaded from %s.", chunks_file)
        return chunks
    except Exception as e:
        logging.error("Error loading chunks: %s", e)
        raise

# Load the state-of-the-art embedding model
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load the fine-tuned LLM with modern device mapping
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
    logging.info("LLM loaded successfully from %s.", LLM_MODEL_NAME)
except Exception as e:
    logging.error("Error loading LLM: %s", e)
    raise

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load FAISS index and chunks
faiss_index = load_faiss_index()
chunks = load_chunks()

def retrieve_relevant_chunks(query: str, top_k: int = 5) -> list:
    """
    Encode the query and retrieve the top_k most similar document chunks from the FAISS index.
    """
    try:
        q_vec = embed_model.encode([query]).astype("float32")
        distances, indices = faiss_index.search(q_vec, top_k)
        top_chunks = [chunks[i] for i in indices[0] if i >= 0]
        logging.info("Retrieved %d chunks for query: '%s'.", len(top_chunks), query)
        return top_chunks
    except Exception as e:
        logging.error("Error during retrieval: %s", e)
        return []

def build_prompt(query: str, context_chunks: list) -> str:
    """
    Build an enriched prompt by injecting the retrieved context before the user query.
    """
    context_text = "\n".join(context_chunks)
    prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    logging.debug("Prompt built: %s", prompt)
    return prompt

def get_rag_answer(query: str, max_length: int = 512) -> str:
    """
    Combine retrieval and generation:
      1. Retrieve the most relevant context chunks using FAISS.
      2. Build a structured prompt with context and query.
      3. Generate an answer using the fine-tuned LLM.
    """
    try:
        top_chunks = retrieve_relevant_chunks(query, top_k=5)
        if not top_chunks:
            logging.warning("No relevant context found; proceeding with query only.")
        prompt = build_prompt(query, top_chunks)
        logging.info("Generating answer with prompt (length %d characters).", len(prompt))
        response = generator(prompt, max_length=max_length, num_return_sequences=1)
        answer = response[0]["generated_text"]
        logging.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logging.error("Error during answer generation: %s", e)
        return "An error occurred during generation."

if __name__ == "__main__":
    user_query = "I want to use the labs SOW template to create an SOW saying what I did this week which was to research potential projects, and that next week I will continue doing that."
    answer = get_rag_answer(user_query)
    print("Answer:\n", answer)
