#!/usr/bin/env python3
"""
index_builder.py

This script builds a FAISS index using the HNSW algorithm (faiss.IndexHNSWFlat) for efficient similarity search.
It attempts GPU indexing only if the index type supports it; otherwise, it falls back to CPU.
"""

import faiss
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def build_faiss_index(embedding_dim: int = 768, index_file: str = "faiss_index.bin"):
    # Load stored chunks and embeddings
    chunks = np.load("chunks.npy", allow_pickle=True)
    embeddings = np.load("embeddings.npy").astype("float32")
    
    logging.info("Building FAISS index using HNSWFlat...")
    # Create an HNSW index with example parameters.
    M = 32
    index = faiss.IndexHNSWFlat(embedding_dim, M)
    index.hnsw.efConstruction = 40

    # Check if GPU conversion is supported for this index type.
    if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
        # HNSWFlat is currently not supported on GPU.
        if index.__class__.__name__ == "IndexHNSWFlat":
            logging.info("HNSWFlat does not support GPU conversion. Using CPU index.")
            index_to_use = index
        else:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            index_to_use = gpu_index
            logging.info("Using GPU for FAISS indexing.")
    else:
        index_to_use = index
        logging.info("Using CPU for FAISS indexing.")

    index_to_use.add(embeddings)
    logging.info("Total embeddings indexed: %d", index_to_use.ntotal)
    
    # If GPU was used and conversion is supported, move the index back to CPU for saving.
    if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources") and index_to_use.__class__.__name__ != "IndexHNSWFlat":
        final_index = faiss.index_gpu_to_cpu(index_to_use)
    else:
        final_index = index_to_use
    
    faiss.write_index(final_index, index_file)
    logging.info("FAISS index built and saved to %s.", index_file)

if __name__ == "__main__":
    build_faiss_index()
