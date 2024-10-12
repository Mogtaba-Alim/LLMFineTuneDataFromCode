import os
import json
import random
import ast
from git import Repo
import networkx as nx
from typing import List, Dict, Any
import anthropic
from dotenv import load_dotenv
import re

# Load .env file
load_dotenv()

# Assume you have set your API key as an environment variable
API_KEY = os.getenv("CLAUDE_API_KEY")

client = anthropic.Anthropic(
    api_key=API_KEY
)

SYSTEM_PROMPT = """
You are an expert programmer and code analyst. Your task is to analyze code snippets and generate high-quality question-answer pairs and code completion tasks. Focus on key programming concepts, best practices, and potential pitfalls. Your responses should be technically accurate, concise, and relevant to improving programming skills.

For question-answer pairs:
- Create questions that test understanding of the code's logic, structure, and potential edge cases.
- Provide clear, informative answers that explain the concepts thoroughly.

For code completion tasks:
- Create partial code snippets that challenge understanding of the original code.
- Ensure the completed versions are correct and follow best practices.
- Make your code as concise but detailed as possible

Maintain a professional and educational tone throughout your responses.


IMPORTANT: Always format your response as a valid JSON array. Each item in the array should be an object with the specified keys.
"""

def call_llm_api(prompt: str) -> str:

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3500,
        temperature=0.2,
        system=SYSTEM_PROMPT
    )

    return response.content[0].text

def fix_json_response(text):
    """
    Takes a text response from the LLM and attempts to return a valid JSON object.
    If the JSON is invalid, it tries to fix common errors and parse again.
    """
    # Remove any text before and after the JSON array/object
    # Find the first '[' and the last ']'
    start = text.find('[')
    end = text.rfind(']') + 1  # Include the closing bracket

    if start != -1 and end != -1:
        json_text = text[start:end]
    else:
        # Cannot find JSON structure in text
        return None

    # Try to parse the text as is
    try:
        data = json.loads(json_text)
        return data
    except json.JSONDecodeError:
        pass  # Proceed to fix the JSON

    # Try to fix common JSON errors
    json_text = fix_common_json_errors(json_text)

    # Try to parse the fixed JSON
    try:
        data = json.loads(json_text)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None  # Parsing failed after attempts to fix

def fix_common_json_errors(text):
    """
    Attempts to fix common JSON errors such as:
    - Single quotes instead of double quotes
    - Unescaped characters within strings
    - Improperly escaped backslashes and quotes
    """
    # Replace single quotes with double quotes where appropriate
    text = re.sub(r"(?<!\\)'", '"', text)

    # Remove trailing commas before closing brackets/braces
    text = re.sub(r',\s*(\]|})', r'\1', text)

    # Escape unescaped double quotes within strings
    def escape_quotes(match):
        content = match.group(0)
        # Escape double quotes inside the string
        content = content[0] + content[1:-1].replace('\\', '\\\\').replace('"', '\\"') + content[-1]
        return content

    text = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', escape_quotes, text)

    # Ensure backslashes are properly escaped
    text = text.replace('\\\\"', '\\\\"')

    # Remove any control characters that are not escaped
    text = ''.join([c if ord(c) >= 32 else '\\n' for c in text])

    return text


def balance_brackets_and_braces(text):
    """
    Ensures that the number of opening and closing brackets/braces are equal.
    """
    brackets = {'[': ']', '{': '}'}
    for open_b, close_b in brackets.items():
        open_count = text.count(open_b)
        close_count = text.count(close_b)
        if open_count > close_count:
            text += close_b * (open_count - close_count)
        elif close_count > open_count:
            text = open_b * (close_count - open_count) + text
    return text

def clone_repo(repo_url: str, target_dir: str) -> str:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(target_dir, repo_name)
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
    return repo_path


def get_code_files(repo_path: str, extensions: tuple = ('.py', '.cpp', ".r", ".R")) -> List[str]:
    code_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(extensions):
                code_files.append(os.path.join(root, file))
    return code_files


def extract_imports_and_dependencies(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, 'r') as file:
        content = file.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"imports": [], "from_imports": []}

    imports = []
    from_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            from_imports.append(f"{node.module}.{node.names[0].name}")

    return {"imports": imports, "from_imports": from_imports}


def generate_qa_pairs(code: str, file: str) -> List[Dict[str, str]]:
    prompt = f"""
    Given the following code snippet, generate 3 question-answer pairs that would be useful for training an LLM to understand and generate similar code:

    {code}

    Format your response as a JSON list of objects, each with 'question' and 'answer' keys.
    Do not include any text, instead have only a json output
    """
    response = call_llm_api(prompt)

    while response[0] != '[':
        response = response[1:]

    if response[-1] != "]":
        if response[-1] != ["}"]:
            if response[-1] != '"':
                response = response + '"}]'
            else:
                response = response + '}]'
        else:
            response = response + "]"

    print(f"Converting Claude Q/A output to JSON for the following file: {file}")

    print(response)
    output_json = fix_json_response(response)

    return output_json


def generate_code_completion_tasks(code: str, file: str) -> List[Dict[str, str]]:
    prompt = f"""
    Given the following code snippet, generate 2 code completion tasks. For each task, provide a partial version of the code and the full completed version. The code should be as efficient as possible meaning it should not use unnecessary statements. 

    {code}

    Format your response as a JSON list of objects, each with 'partial' and 'complete' keys.
    IMPORTANT: Always format your response as a valid JSON array. Each item in the array should be an object with the specified keys.
    
    Do not output any text!
    """
    response = call_llm_api(prompt)

    while response[0] != '[':
        response = response[1:]

    if response[-1] != "]":
        if response[-1] != ["}"]:
            if response [-1] != '"':
                response = response + '"}]'
            else:
                response = response + '}]'
        else:
            response = response + "]"

    print(f"Converting Claude Completion output to JSON for the following file: {file}")

    print(response)
    output_json = fix_json_response(response)

    return output_json


def create_dataset_entry(repo_url: str, file_path: str, content: str, language: str) -> Dict[str, Any]:
    qa_pairs = generate_qa_pairs(content, os.path.relpath(file_path, repo_url))
    completion_tasks = generate_code_completion_tasks(content, os.path.relpath(file_path, repo_url))
    dependencies = extract_imports_and_dependencies(file_path)

    return {
        'repo': repo_url,
        'file': os.path.relpath(file_path, repo_url),
        'language': language,
        'content': content,
        'qa_pairs': qa_pairs,
        'completion_tasks': completion_tasks,
        'dependencies': dependencies
    }


def analyze_project_dependencies(repo_path: str) -> Dict[str, List[str]]:
    G = nx.DiGraph()
    file_to_module = {}

    for file_path in get_code_files(repo_path):
        module_name = os.path.relpath(file_path, repo_path).replace('/', '.').replace('.py', '')
        file_to_module[file_path] = module_name
        G.add_node(module_name)

        deps = extract_imports_and_dependencies(file_path)
        for imp in deps['imports'] + deps['from_imports']:
            if imp in file_to_module.values():
                G.add_edge(module_name, imp)

    return {module: list(G.successors(module)) for module in G.nodes()}


def create_dataset(repo_urls: List[str], output_file: str, target_dir: str = 'repos'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    dataset = []

    for repo_url in repo_urls:
        print(repo_url)
        repo_path = clone_repo(repo_url, target_dir)
        code_files = get_code_files(repo_path)
        project_dependencies = analyze_project_dependencies(repo_path)

        for file_path in code_files:
            with open(file_path, 'r') as file:
                content = file.read()

            language = os.path.splitext(file_path)[1][1:]
            entry = create_dataset_entry(repo_url, file_path, content, language)

            # Add project-level dependencies
            module_name = os.path.relpath(file_path, repo_path).replace('/', '.').replace('.py', '')
            entry['project_dependencies'] = project_dependencies.get(module_name, [])

            dataset.append(entry)

    # Shuffle the dataset
    random.shuffle(dataset)

    # Split into train and validation sets (80/20 split)
    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

    # Write to JSON files
    with open(f'{output_file}_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(f'{output_file}_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)


# Example usage
repo_urls = [
    # 'https://github.com/bhklab/PharmacoGx.git',
    # 'https://github.com/bhklab/med-imagetools.git',
    # 'https://github.com/bhklab/AnnotationGx.git',
    # 'https://github.com/bhklab/readii.git',
    # 'https://github.com/bhklab/PymRMRe.git', # Skipping for now due to expt files
    # 'https://github.com/bhklab/mRMRe.git',
    # 'https://github.com/bhklab/CoreGx.git',
    # 'https://github.com/bhklab/RadioGx.git',
    'https://github.com/bhklab/genefu.git',
    'https://github.com/bhklab/survcomp.git',
    'https://github.com/bhklab/ToxicoGx.git'
]

create_dataset(repo_urls, 'advanced_code_dataset3')