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

API_KEY = os.getenv("CLAUDE_API_KEY")

client = anthropic.Anthropic(api_key=API_KEY)

##############################################################################
#                         IMPROVED SYSTEM PROMPT
##############################################################################
SYSTEM_PROMPT = """
You are an expert programmer and code analyst. Your task is to analyze code snippets and generate diverse, high-quality training data with perfect JSON output. 

Focus on tasks such as:
  - Q&A pairs (covering logic, structure, edge cases).
  - Code completion tasks.
  - Debugging tasks (finding or explaining potential bugs).
  - Refactoring tasks (improving performance or readability).
  - Docstring generation tasks (demonstrating best practices for documenting the code).

**CRUCIAL**:
1. Always produce strictly valid JSON arrays or objects—no extra text, no explanations, no markdown formatting.
2. If you must produce multiple items, wrap them in a JSON array, e.g., `[ {...}, {...} ]`.
3. Do not output text before the opening bracket `[` or after the closing bracket `]`.
4. If you need to reference code, do so as strings inside the JSON. 
5. If the content is large, you can abbreviate code within reason, but do not break JSON structure.
6. If you are given partial JSON to fix or complete, only output the corrected JSON—do not add extra commentary.

Follow these rules strictly to avoid invalid JSON or partial responses.
"""

##############################################################################
#                  CALL CLAUDE API WITH A GIVEN PROMPT
##############################################################################
def call_llm_api(prompt: str) -> str:
    """
    Generic call to the Claude API with a given user prompt.
    Uses a shared SYSTEM_PROMPT to ensure instructions are consistent.
    Returns raw text from Claude's response.
    """
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3500,   # You can adjust this if you prefer
        temperature=0.2,
        system=SYSTEM_PROMPT
    )
    return response.content[0].text

##############################################################################
#                PARSING AND FIXING JSON RESPONSES
##############################################################################
def fix_json_response(text: str) -> Any:
    """
    Attempts to parse a text response from the LLM as valid JSON.
    1) Extract from first '[' to the last ']'.
    2) Attempt direct json.loads.
    3) If fail, do local text fix and re-parse.
    4) If still fail, do a second call to the LLM with the invalid JSON requesting a fix.
    5) Return a Python object if successful, else None.
    """
    print("\n=== Attempting to parse LLM response as JSON ===")

    # Step A: Try to isolate bracketed content
    print("Step A: Isolating potential JSON text from response...")
    start = text.find('[')
    end = text.rfind(']') + 1

    if start == -1 or end == -1 or start >= end:
        print("Could not find '[' and ']' in the right order. Checking for object-based JSON '{...}' instead.")
        start_alt = text.find('{')
        end_alt = text.rfind('}') + 1
        if start_alt != -1 and end_alt != -1 and start_alt < end_alt:
            json_text = text[start_alt:end_alt]
            print("Found curly-brace style JSON snippet.")
        else:
            print("No bracket or brace style JSON found. Returning None.")
            return None
    else:
        json_text = text[start:end]
        print("Found bracketed JSON snippet.")

    # Step B: Try direct parse
    print("Step B: Attempting direct parse of the raw snippet...")
    try:
        data = json.loads(json_text)
        print("Success: Direct parse worked!")
        return data
    except json.JSONDecodeError:
        print("Direct parse failed. Moving to local fix step.")

    # Step C: Attempt local fixes
    print("Step C: Applying local JSON fixes (quotes, commas, etc.)...")
    json_text_fixed = local_json_fixes(json_text)
    try:
        data = json.loads(json_text_fixed)
        print("Success: Local fix parse worked!")
        return data
    except json.JSONDecodeError:
        print("Local fix parse also failed.")

    # Step D: Re-call the LLM to fix invalid JSON snippet
    print("Step D: Re-calling LLM to fix invalid JSON snippet...")
    fix_prompt = f"""
    You provided an invalid or partial JSON snippet. Please output a corrected valid JSON.
    Remember: no extra text, just the corrected JSON.

    Invalid snippet:
    {json_text_fixed}
    """
    corrected = call_llm_api(fix_prompt).strip()

    print("Received second pass from LLM. Attempting to parse corrected JSON...")

    # Now parse the second pass
    start2 = corrected.find('[')
    end2 = corrected.rfind(']') + 1
    if start2 == -1 or end2 == -1 or start2 >= end2:
        # fallback to curly braces check
        start_alt2 = corrected.find('{')
        end_alt2 = corrected.rfind('}') + 1
        if start_alt2 != -1 and end_alt2 != -1 and start_alt2 < end_alt2:
            corrected_json = corrected[start_alt2:end_alt2]
        else:
            print("Could not parse second pass text as bracket or brace JSON. Returning None.")
            return None
    else:
        corrected_json = corrected[start2:end2]

    corrected_json = local_json_fixes(corrected_json)
    try:
        data = json.loads(corrected_json)
        print("Success: Second pass LLM-fix parse worked!")
        return data
    except json.JSONDecodeError:
        print("Second pass JSON parse also failed. Returning None.")
        return None

def local_json_fixes(text: str) -> str:
    """
    Apply several heuristics to correct common JSON errors:
      - Single quotes -> double quotes
      - Remove trailing commas
      - Escape unescaped quotes inside strings
      - Balance brackets/braces
    """
    print("  [local_json_fixes] Balancing brackets/braces...")
    text = balance_brackets_and_braces(text)

    print("  [local_json_fixes] Replacing single quotes with double quotes if possible...")
    text = re.sub(r'(?<!\\)\'', '"', text)

    print("  [local_json_fixes] Removing trailing commas...")
    text = re.sub(r',\s*(\]|})', r'\1', text)

    print("  [local_json_fixes] Escaping quotes inside strings...")
    def escape_quotes(match):
        # match.group(0) is the entire string including quotes
        s = match.group(0)
        # inside part (excluding the wrapping quotes)
        inside = s[1:-1]
        # escape double quotes and backslashes inside
        inside = inside.replace('\\', '\\\\').replace('"', '\\"')
        return '"' + inside + '"'

    text = re.sub(r'(["\'])(?:(?=(\\?))\2.)*?\1', escape_quotes, text)

    print("  [local_json_fixes] Removing unprintable control characters...")
    cleaned = []
    for c in text:
        if ord(c) < 32 and c not in ['\n', '\t']:
            continue
        cleaned.append(c)
    text = "".join(cleaned)

    return text

def balance_brackets_and_braces(text: str) -> str:
    """
    Ensures that the number of opening and closing brackets/braces are equal.
    If there's an imbalance, attempts to fix by adding missing ones at the end.
    """
    pairs = [
        ('[', ']'),
        ('{', '}')
    ]
    for open_b, close_b in pairs:
        diff = text.count(open_b) - text.count(close_b)
        if diff > 0:
            text += close_b * diff
        elif diff < 0:
            text = (open_b * abs(diff)) + text
    return text

##############################################################################
#                           GIT & DEPENDENCIES
##############################################################################
def clone_repo(repo_url: str, target_dir: str) -> str:
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    repo_path = os.path.join(target_dir, repo_name)
    if not os.path.exists(repo_path):
        Repo.clone_from(repo_url, repo_path)
    return repo_path

def get_code_files(repo_path: str, extensions: tuple = ('.py', '.cpp', '.r', '.R')) -> List[str]:
    code_files = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(extensions):
                code_files.append(os.path.join(root, file))
    return code_files

def extract_imports_and_dependencies(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
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

##############################################################################
#                    GENERATION TASKS (Q&A, COMPLETIONS, ETC.)
##############################################################################

#
# 1) Q&A Pairs
#
def generate_qa_pairs(code: str, file: str) -> List[Dict[str, str]]:
    prompt = f"""
    Given the following code snippet, generate 3 question-answer pairs that would be useful 
    for training an LLM to understand and generate similar code.

    Code snippet:
    {code}

    **Return only valid JSON** in the format:
    [
      {{"question": "...", "answer": "..."}},
      ...
    ]
    No extra text or explanation. 
    """
    print(f"\n[generate_qa_pairs] Sending Q&A prompt for file: {file}")
    response = call_llm_api(prompt).strip()
    output_json = fix_json_response(response)
    if output_json is None:
        print("[generate_qa_pairs] Final: Could not parse valid JSON for Q&A.")
    else:
        print("[generate_qa_pairs] Final: Successfully parsed Q&A JSON!")
    return output_json if output_json else []

#
# 2) Code Completion Tasks
#
def generate_code_completion_tasks(code: str, file: str) -> List[Dict[str, str]]:
    prompt = f"""
    Given the following code snippet, generate 2 code completion tasks. 
    For each task, provide a partial version of the code and the full completed version.
    The code should be as concise as possible, no unnecessary statements.

    Code snippet:
    {code}

    **Return only valid JSON** in this format:
    [
      {{"partial": "...", "complete": "..."}},
      ...
    ]
    No extra text or explanation.
    """
    print(f"\n[generate_code_completion_tasks] Sending code completion prompt for file: {file}")
    response = call_llm_api(prompt).strip()
    output_json = fix_json_response(response)
    if output_json is None:
        print("[generate_code_completion_tasks] Final: Could not parse valid JSON for code completions.")
    else:
        print("[generate_code_completion_tasks] Final: Successfully parsed completions JSON!")
    return output_json if output_json else []

#
# 3) Debugging Tasks
#
def generate_debugging_tasks(code: str, file: str) -> List[Dict[str, str]]:
    """
    Generate tasks that identify or explain potential bugs in the snippet.
    Each item: { "bug_description": "", "bug_fix": "" }
    """
    prompt = f"""
    Analyze this code for possible bugs or bad practices and generate 2 'debugging tasks'. 
    Each should have:
      "bug_description"
      "bug_fix"

    Code snippet:
    {code}

    **Return only valid JSON** in the format:
    [
      {{"bug_description": "...", "bug_fix": "..."}},
      ...
    ]
    """
    print(f"\n[generate_debugging_tasks] Sending debugging prompt for file: {file}")
    response = call_llm_api(prompt).strip()
    output_json = fix_json_response(response)
    if output_json is None:
        print("[generate_debugging_tasks] Final: Could not parse valid JSON for debugging tasks.")
    else:
        print("[generate_debugging_tasks] Final: Successfully parsed debugging JSON!")
    return output_json if output_json else []

#
# 4) Refactoring Tasks
#
def generate_refactoring_tasks(code: str, file: str) -> List[Dict[str, str]]:
    """
    Generate 2 refactoring ideas:
      { "original_snippet": "...", "refactored_snippet": "...", "explanation": "..." }
    """
    prompt = f"""
    Given this code snippet, propose 2 refactoring ideas (improving performance or readability).
    Each object:
      "original_snippet"
      "refactored_snippet"
      "explanation"

    Code snippet:
    {code}

    **Return only valid JSON** in the format:
    [
      {{"original_snippet": "...", "refactored_snippet": "...", "explanation": "..."}},
      ...
    ]
    """
    print(f"\n[generate_refactoring_tasks] Sending refactoring prompt for file: {file}")
    response = call_llm_api(prompt).strip()
    output_json = fix_json_response(response)
    if output_json is None:
        print("[generate_refactoring_tasks] Final: Could not parse valid JSON for refactoring tasks.")
    else:
        print("[generate_refactoring_tasks] Final: Successfully parsed refactoring JSON!")
    return output_json if output_json else []

#
# 5) Docstring Generation Tasks
#
def generate_docstring_tasks(code: str, file: str) -> List[Dict[str, str]]:
    """
    For each docstring: { "function_signature": "...", "docstring": "..." }
    """
    prompt = f"""
    Identify 1-2 functions or classes in this code snippet and propose docstrings. 
    Each object in JSON: 
      "function_signature": "...",
      "docstring": "..."

    Code snippet:
    {code}

    **Return only valid JSON** in the format:
    [
      {{"function_signature": "...", "docstring": "..."}},
      ...
    ]
    """
    print(f"\n[generate_docstring_tasks] Sending docstring prompt for file: {file}")
    response = call_llm_api(prompt).strip()
    output_json = fix_json_response(response)
    if output_json is None:
        print("[generate_docstring_tasks] Final: Could not parse valid JSON for docstring tasks.")
    else:
        print("[generate_docstring_tasks] Final: Successfully parsed docstring JSON!")
    return output_json if output_json else []

##############################################################################
#                       CREATE DATASET ENTRY
##############################################################################
def create_dataset_entry(repo_url: str, file_path: str, content: str, language: str) -> Dict[str, Any]:
    """
    Gathers multiple data types for each code snippet:
      - Q&A
      - Code completion
      - Debugging tasks
      - Refactoring tasks
      - Docstring tasks
    """
    # Basic tasks
    qa_pairs = generate_qa_pairs(content, os.path.relpath(file_path, repo_url))
    completion_tasks = generate_code_completion_tasks(content, os.path.relpath(file_path, repo_url))

    # Additional tasks for variety
    debugging_tasks = generate_debugging_tasks(content, os.path.relpath(file_path, repo_url))
    refactoring_tasks = generate_refactoring_tasks(content, os.path.relpath(file_path, repo_url))
    docstring_tasks = generate_docstring_tasks(content, os.path.relpath(file_path, repo_url))

    # Dependencies
    dependencies = extract_imports_and_dependencies(file_path)

    return {
        'repo': repo_url,
        'file': os.path.relpath(file_path, repo_url),
        'language': language,
        'content': content,
        'qa_pairs': qa_pairs,
        'completion_tasks': completion_tasks,
        'debugging_tasks': debugging_tasks,
        'refactoring_tasks': refactoring_tasks,
        'docstring_tasks': docstring_tasks,
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

##############################################################################
#                           CREATE DATASET
##############################################################################
def create_dataset(repo_urls: List[str], output_file: str, target_dir: str = 'repos'):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    dataset = []

    for repo_url in repo_urls:
        print(f"\n=== Processing repository: {repo_url} ===")
        repo_path = clone_repo(repo_url, target_dir)
        code_files = get_code_files(repo_path)
        project_dependencies = analyze_project_dependencies(repo_path)

        for file_path in code_files:
            print(f"\n--- Analyzing code file: {file_path} ---")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

            language = os.path.splitext(file_path)[1][1:]
            entry = create_dataset_entry(repo_url, file_path, content, language)

            # Add project-level dependencies
            module_name = os.path.relpath(file_path, repo_path).replace('/', '.').replace('.py', '')
            entry['project_dependencies'] = project_dependencies.get(module_name, [])

            dataset.append(entry)

    # Shuffle the dataset
    random.shuffle(dataset)

    # 80/20 split
    split_index = int(len(dataset) * 0.8)
    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

    print(f"\nSaving dataset to {output_file}_train.json and {output_file}_val.json")
    with open(f'{output_file}_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)

    with open(f'{output_file}_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)

# Example usage
repo_urls = [
    'https://github.com/bhklab/PharmacoGx.git',
    'https://github.com/bhklab/med-imagetools.git',
    'https://github.com/bhklab/AnnotationGx.git',
    # 'https://github.com/bhklab/readii.git',
    # 'https://github.com/bhklab/PymRMRe.git', # Possibly skip if it has large files
    # 'https://github.com/bhklab/mRMRe.git',
    # 'https://github.com/bhklab/CoreGx.git',
    # 'https://github.com/bhklab/RadioGx.git',
    # 'https://github.com/bhklab/genefu.git',
    # 'https://github.com/bhklab/survcomp.git',
    # 'https://github.com/bhklab/ToxicoGx.git'
]

create_dataset(repo_urls, 'advanced_code_dataset1')
