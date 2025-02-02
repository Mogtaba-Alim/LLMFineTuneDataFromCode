import faiss
import numpy as np

def build_faiss_gpu_index(embedding_dim: int = 384, index_file="faiss_index.bin"):
    """Build and save a GPU-enabled FAISS index."""
    # 1. Load chunks & embeddings
    chunks = np.load("chunks.npy", allow_pickle=True)
    embeddings = np.load("embeddings.npy").astype('float32')

    # 2. Build CPU index, then move to GPU
    cpu_index = faiss.IndexFlatL2(embedding_dim)
    res = faiss.StandardGpuResources()  # GPU resource
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    
    # 3. Add embeddings on GPU
    gpu_index.add(embeddings)

    # 4. Transfer index back to CPU for saving
    final_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(final_index, index_file)

if __name__ == "__main__":
    build_faiss_gpu_index()
    print("GPU FAISS index built and saved.")

