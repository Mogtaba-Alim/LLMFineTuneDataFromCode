# LLMFineTuneDataFromCode

[![License](https://img.shields.io/github/license/Mogtaba-Alim/LLMFineTuneDataFromCode)](LICENSE)

LLMFineTuneDataFromCode is a Python package designed to streamline the process of generating fine-tuning datasets for large language models (LLMs). It leverages code repositories and research papers to create datasets and provides tools to train and run inference on fine-tuned LLMs. 

This package is tailored for labs and organizations aiming to extract insights from their existing resources for LLM training while ensuring flexibility and extensibility.

---

## Features

### 1. Data Generation Pipeline
- **Code-Based Dataset Generation**:
  - The `generateCodeFineTune.py` script processes GitHub repositories to generate training datasets from code files. 
  - It includes dependency analysis, Q&A generation, and code completion tasks.
- **Research Paper Dataset Generation**:
  - The `createFinalDataOutput.py` script processes research papers in PDF format to generate Q&A pairs. 
  - Questions can be customized by editing the `QUESTIONS` list.

### 2. LLM Fine-Tuning Tools
- **Training**:
  - Use the `train.ipynb` script to fine-tune an LLM with the generated dataset. 
  - Tested with Meta’s Llama 3.1 8B model (requires access from Huggingface).
- **Inference**:
  - Run inference on the fine-tuned model using the `inference.ipynb` script to interactively query your model.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mogtaba-Alim/LLMFineTuneDataFromCode.git
   cd LLMFineTuneDataFromCode
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file and include your API keys (e.g., OpenAI or Anthropic) as follows:
     ```
     OPENAI_KEY=your_openai_api_key
     CLAUDE_API_KEY=your_claude_api_key
     HF_KEY=your_huggingface_key
     ```

4. Ensure GPU resources are available and compatible. The scripts are tested with NVIDIA A100 GPUs (40GB VRAM) and CUDA 12.6.

---

## Usage

### 1. Generate a Fine-Tuning Dataset
#### From Code Repositories
```bash
python generateCodeFineTune.py
```
- Specify GitHub repository links in the script.

#### From Research Papers
```bash
python createFinalDataOutput.py
```
- Place your research papers (PDFs) in the `./Papers` folder. 
- Customize the questions in the `QUESTIONS` list as needed.

### 2. Fine-Tune an LLM
```bash
python train.ipynb
```
- Update the training script with your Huggingface model and dataset paths.

### 3. Run Inference
```bash
python inference.ipynb
```
- Interactively query your fine-tuned model.

---

## Requirements

- Python 3.8+
- CUDA 12.6
- Huggingface Transformers
- PyTorch
- Other dependencies listed in `requirements.txt`

---

## Notes

- **GPU Resources**: The package is optimized for high-memory GPUs like A100. Adjust parameters in `train.ipynb` for smaller GPUs.
- **Access Requirements**: You must have a Huggingface account and request access to the model being fine-tuned (e.g., Meta's Llama 3.1 8B).

---

## Contribution

We welcome contributions to improve the package! Please submit a pull request or create an issue to share your ideas or report bugs.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This package was developed by Mogtaba Alim to facilitate efficient dataset generation and LLM fine-tuning. Special thanks to the open-source community for their invaluable tools and libraries.

---

Feel free to replace or expand on any section as needed for your specific requirements. Let me know if you'd like to refine this further!