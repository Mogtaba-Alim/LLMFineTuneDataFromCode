import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.npy"
EMBS_DIM = 384  # Must match the embeddings dimension

# Load FAISS index & chunks
index = faiss.read_index(INDEX_FILE)
chunks = np.load(CHUNKS_FILE, allow_pickle=True)

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load finetuned LLM from HF (replace with your model)
LLM_MODEL_NAME = "path-or-hub-username/finetuned-lab-model"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def retrieve_relevant_chunks(query: str, top_k: int = 3):
    """Embed the query and retrieve top-k similar chunks."""
    q_vec = embed_model.encode([query])
    q_vec = q_vec.astype('float32')
    distances, indices = index.search(q_vec, top_k)
    top_chunks = [chunks[i] for i in indices[0]]
    return top_chunks

def build_prompt(query: str, context_chunks: list) -> str:
    """Inject the retrieved chunks into the prompt."""
    context_text = "\n".join(context_chunks)
    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    return prompt

def get_rag_answer(query: str) -> str:
    """Combine retrieval with LLM inference."""
    top_chunks = retrieve_relevant_chunks(query, top_k=5)
    prompt = build_prompt(query, top_chunks)
    response = generator(prompt, max_length=512, num_return_sequences=1)
    return response[0]["generated_text"]

if __name__ == "__main__":
    # Example usage:
    user_query = "How do I use the lab presentation template for a new project?"
    answer = get_rag_answer(user_query)
    print("Answer:\n", answer)
