import json
import os
import random
from typing import List, Dict
from openai import OpenAI
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import re
import unicodedata
from dotenv import load_dotenv

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading necessary NLTK data...")
        nltk.download('punkt_tab', quiet=True)

print("Ensuring NLTK data...")
ensure_nltk_data()
print("NLTK data check complete.")

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(api_key=OPENAI_KEY)
print("OpenAI client initialized.")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_whitespace(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))
    return text.strip()

def normalize_unicode(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def split_long_sentences(text, max_length=100):
    sentences = sent_tokenize(text)
    new_sentences = []
    for sentence in sentences:
        if len(sentence) > max_length:
            parts = re.split(r'[,;:]', sentence)
            new_parts = []
            current_part = ''
            for part in parts:
                if len(current_part) + len(part) < max_length:
                    current_part += part + (', ' if current_part else '')
                else:
                    if current_part:
                        new_parts.append(current_part.strip())
                    current_part = part + ', '
            if current_part:
                new_parts.append(current_part.strip())
            new_sentences.extend(new_parts)
        else:
            new_sentences.append(sentence)
    return ' '.join(new_sentences)

def clean_research_paper(text):
    text = clean_whitespace(text)
    text = normalize_unicode(text)
    text = split_long_sentences(text)
    return text

def chunk_text(text: str, chunk_size: int = 2048, chunk_overlap: int = 200) -> List[str]:
    """
    Splits `text` into overlapping chunks. 
    chunk_size and chunk_overlap are approximate 
    (word-based) in this demo.
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunk_str = " ".join(chunk)
        chunks.append(chunk_str)
        
        # Move start forward by (chunk_size - overlap)
        # ensuring we don't get stuck or go backward.
        # Also avoid negative or zero step.
        step = max(chunk_size - chunk_overlap, 1)
        start += step
    
    return chunks

# Original question list
QUESTIONS = [
    "What is the main objective of the research in this paper?",
    "Can you summarize the abstract of the paper?",
    "What are the softwares and computational tools that were used in this paper?",
    "Describe the methodology used in the paper.",
    "What are the key findings of the paper?",
    "How was the data analyzed in the study?",
    "Was the data in the study pre-processed in anyway? If so how?",
    "What conclusions were drawn in the paper?",
    "Can you provide a summary of the literature review from the paper?",
    "What future research directions do the authors suggest in the paper?",
    "What statistical techniques were used in the paper?",
    "Can you describe the experimental setup in the paper?",
    "What are the implications of the research findings?",
    "What are the limitations and delimitations mentioned in the paper?",
    "What recommendations do the authors make in the paper?",
    "Who funded the research in the paper?",
    "Is there any conflict of interest disclosed in the paper?",
    "What ethical considerations are discussed in the paper?",
    "Which studies are most frequently cited in the paper?",
    "Can you explain the technical terms used in the paper?",
    "What data sources were used in the paper, and are they accessible for further research?",
    "Can you summarize the research paper?",
    "What is the significance of this research in its field?",
    "How does this paper contribute to the existing body of knowledge?",
    "Are there any novel techniques or approaches introduced in this paper?",
    "What are the potential real-world applications of this research?",
    "How does the paper address potential biases in the research?",
    "What validation methods were used to ensure the reliability of the results?",
    "Are there any contradictions between this paper's findings and previous research?",
    "What are the main limitations of the study?",
    "What are the main strengths of the study?",
    "What are the main implications of the study?",
    "What are the main recommendations for future research?",
    "What are the main conclusions of the study?",
    "What are the main findings of the study?",
    "What are the main results of the study?",
]

def get_answer_from_gpt_with_title(pdf_title: str, question: str, chunk: str) -> str:
    """
    Calls the LLM API with additional instructions:
    1. The chunk is part of a research paper titled `pdf_title`.
    2. If the chunk doesn't contain the relevant info, 
       the LLM should say: 'the answer to this question is not included in the chunk'.
    """
    prompt = (
        f"You are a research assistant with expertise in academic papers.\n"
        f"This text is from a paper titled: {pdf_title}.\n\n"
        f"Chunk content:\n\"\"\"{chunk}\"\"\"\n\n"
        f"Question: {question}\n\n"
        f"IMPORTANT: If the answer is not clearly found in the chunk, "
        f"please respond with: 'the answer to this question is not included in the chunk'."
    )
    
    print(f"Generating answer for question: {question[:50]}...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or your actual LLM endpoint
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.2,
        response_format={"type": "text"}
    )
    print("Answer generated.")
    
    return response.choices[0].message.content.strip()


def generate_qa_pairs(pdf_title: str, paper_chunks: List[str]) -> List[Dict[str, str]]:
    print("Generating QA pairs across all chunks...")
    
    qa_pairs = []
    chunk_id = 0
    
    for chunk in paper_chunks:
        chunk_id += 1
        print(f"Processing chunk {chunk_id} out of {len(paper_chunks)}...")
        
        for i, question in enumerate(QUESTIONS):
            print(f"  Q{i+1}/{len(QUESTIONS)}: {question[:50]}...")
            
            # We call get_answer_from_gpt with the chunk and the question
            # Add a prompt that instructs the LLM to say “answer not included” if missing.
            answer = get_answer_from_gpt_with_title(pdf_title, question, chunk)
            
            qa_pairs.append({
                "chunk": chunk,
                "question": question,
                "answer": answer
            })
    
    print("QA pairs generation complete for all chunks.")
    return qa_pairs


def process_single_paper(pdf_path: str, output_dir: str) -> Dict:
    filename = os.path.basename(pdf_path)
    output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output.json")
    
    if os.path.exists(output_file):
        print(f"Output for {filename} already exists. Skipping...")
        with open(output_file, 'r') as f:
            return json.load(f)
    
    print(f"Processing file: {filename}")
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_research_paper(pdf_text)
    
    # ----------------- NEW: Chunk the text here ------------------
    paper_chunks = chunk_text(cleaned_text, chunk_size=2048, chunk_overlap=200)
    # -------------------------------------------------------------
    
    print("Generating QA pairs for the paper...")
    # Pass the filename (title) and the list of chunks to the Q&A generator
    paper_data = {
        "repo": "research_papers",
        "file": filename,
        "language": "research_paper",
        "qa_pairs": generate_qa_pairs(filename, paper_chunks)  # <---- updated
    }
    
    print(f"Saving output for {filename}")
    with open(output_file, 'w') as f:
        json.dump(paper_data, f, indent=2)
    
    print(f"Finished processing {filename}")
    return paper_data


def process_research_papers(papers_dir: str, output_dir: str) -> List[Dict]:
    print(f"Processing research papers from directory: {papers_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    research_data = []
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(papers_dir, filename)
            paper_data = process_single_paper(pdf_path, output_dir)
            research_data.append(paper_data)
    
    print("All papers processed.")
    return research_data

def combine_paper_outputs(output_dir: str) -> List[Dict]:
    print("Combining individual paper outputs...")
    combined_data = []
    if not os.path.exists(output_dir):
        #create the output directory
        os.makedirs(output_dir)
    for filename in os.listdir(output_dir):
        if filename.endswith("_output.json"):
            with open(os.path.join(output_dir, filename), 'r') as f:
                combined_data.append(json.load(f))
    return combined_data

def split_data(data: List[Dict], train_ratio: float = 0.8) -> tuple:
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

def save_json(data: List[Dict], filename: str):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    papers_dir = "./paper_Input"
    output_dir = "./Paper_Outputs_with_chunks"
    print(f"Starting to process papers from {papers_dir}")
    
    process_research_papers(papers_dir, output_dir)
    
    research_data = combine_paper_outputs(output_dir)
    
    print("Splitting data into training and validation sets")
    train_data, val_data = split_data(research_data)
    
    print("Loading existing code data")
    with open("code_combined_train_dataset.json", "r") as f:
        code_train_data = json.load(f)
    with open("code_combined_val_dataset.json", "r") as f:
        code_val_data = json.load(f)
    
    print("Combining code and research paper data")
    combined_train_data = code_train_data + train_data
    combined_val_data = code_val_data + val_data
    
    print("Saving combined data")
    save_json(combined_train_data, "combined_dataset_train.json")
    save_json(combined_val_data, "combined_dataset_val.json")

    print("Training data saved to combined_dataset_train.json")
    print("Validation data saved to combined_dataset_val.json")
    print("Process completed.")
