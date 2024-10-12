import os
import openai
import requests
import pandas as pd
import dotenv

dotenv.load_dotenv()

# Setup the API keys
OPENAI_KEY = os.getenv("OPENAI_KEY")
COHERE_KEY = os.getenv("COHERE_KEY")

# Define paths
input_dir = './Papers'
output_file = 'Paper_finetuning_dataset.csv'

# Define your questions
questions = [
    "What is the main objective of the research in this paper?",
    "Summarize the abstract of the paper.",
    "What are the softwares and computational tools that were used in this paper?",
    "Describe the methodology used in the paper.",
    "What are the key findings of the paper?",
    "How was the data analyzed in the study?",
    "Was the data in the study pre-processed in anyway? If so how?",
    "What conclusions were drawn in the paper?",
    "Provide a summary of the literature review from the paper.",
    "What future research directions do the authors suggest in the paper?",
    "What statistical techniques were used in the paper?",
    "Describe the experimental setup in the paper.",
    "What are the implications of the research findings?",
    "What are the limitations and delimitations mentioned in the paper?",
    "What recommendations do the authors make in the paper?",
    "Who funded the research in the paper?",
    "Is there any conflict of interest disclosed in the paper?",
    "What ethical considerations are discussed in the paper?",
    "Which studies are most frequently cited in the paper?",
    "Explain the technical terms used in the paper.",
    "What data sources were used in the paper, and are they accessible for further research?"
]


def extract_text_from_pdf(pdf_path):
    import fitz
    doc = fitz.open(pdf_path)
    text = ''
    for page in doc:
        text += page.get_text()
    return text


def split_text(text, max_tokens=1500):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0
    for word in words:
        word_tokens = len(word) // 4 + 1
        if current_tokens + word_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(word)
        current_tokens += word_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def call_openai_api(prompt, model="gpt-3.5-turbo-0125"):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def call_cohere_api(prompt):
    headers = {
        'Authorization': f'Bearer {COHERE_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'command-r-plus',
        'prompt': prompt
    }
    response = requests.post('https://api.cohere.ai/generate', headers=headers, json=data)
    return response.json()['text'].strip()


total_number_of_chunks = 0
total_number_number_of_requests = 0
average_chunk_size = 0


def generate_finetuning_data(input_dir, questions, output_file, api_choice):
    data = []
    count = 0
    for filename in os.listdir(input_dir):
        print("This is paper number:" + str(count))
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            paper_text = extract_text_from_pdf(pdf_path)
            paper_chunks = split_text(paper_text)
            global total_number_of_chunks
            total_number_of_chunks += len(paper_chunks)
            global total_number_number_of_requests
            total_number_number_of_requests += len(paper_chunks) * len(questions)
            print("Length of chunks:" + str(len(paper_chunks)))
            print("Number of questions" + str(len(questions)))
            print("Total = " + str(len(paper_chunks) * len(questions)))
            for chunk in paper_chunks:
                for question in questions:
                    prompt = f"Research Paper: {chunk}\n\nQuestion: {question}. Make sure your answers are straight forward and sticks to premise of the question without too much speculation where every answer should be from the paper only."
                    print("ANOTHER ONE!!!!!!!")
                    print(count)
                    if api_choice == (1, 0, 0):
                        answer = call_openai_api(prompt, model="gpt-3.5-turbo")
                    elif api_choice == (0, 1, 0):
                        answer = call_openai_api(prompt, model="gpt-4o")
                    elif api_choice == (0, 0, 1):
                        answer = call_cohere_api(prompt)
                    else:
                        raise ValueError(
                            "Invalid API choice. The one-hot vector must be one of (1, 0, 0), (0, 1, 0), or (0, 0, 1).")
                    print(answer)
                    data.append({
                        'filename': filename,
                        'question': question,
                        'chunk': chunk,
                        'answer': answer
                    })
        count += 1
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# Example of running the pipeline
api_choice = (1, 0, 0)  # Choose the API: (GPT-3.5, GPT-4o, Cohere)
generate_finetuning_data(input_dir, questions, output_file, api_choice)
print("This is tht total number of chunks: " + str(total_number_of_chunks))
print("This is the total number of requests: " + str(total_number_number_of_requests) )
