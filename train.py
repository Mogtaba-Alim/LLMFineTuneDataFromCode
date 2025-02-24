from huggingface_hub import notebook_login
notebook_login()

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling  # Add this for better data handling
)
from trl import SFTTrainer
import os
from typing import Tuple
import logging
from transformers import GenerationConfig
from time import perf_counter
import torch
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_id="meta-llama/Meta-Llama-3.1-8B"

def get_model_and_tokenizer(
    model_id: str,
    max_seq_length: int = 2048,  # Match with trainer config
    load_in_8bit: bool = False,  # Option for 8-bit quantization
    trust_remote_code: bool = True  # Important for some models
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and configure model and tokenizer with enhanced error handling and options.

    Args:
        model_id: HuggingFace model identifier
        max_seq_length: Maximum sequence length for model/tokenizer
        load_in_8bit: Whether to use 8-bit quantization instead of 4-bit
        trust_remote_code: Whether to trust remote code in model files
    """
    try:
        # Initialize tokenizer with safety settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            model_max_length=max_seq_length,
            padding_side="right",  # Consistent padding
            truncation_side="right",  # Consistent truncation
        )

        # Handle special tokens more robustly
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.mask_token = tokenizer.eos_token

        # Configure quantization
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_threshold=6.0,
            )

        # Load model with enhanced settings
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16,
            use_flash_attention_2=False,  # Explicitly disable Flash Attention
            use_cache=True  # Enable KV cache for inference
        )

        # Configure model settings
        model.config.pretraining_tp = 1
        model.config.torch_dtype = torch.float16

        # Add model configuration for better training
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.max_length = max_seq_length

        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Log model and tokenizer configuration
        logger.info(f"Model loaded: {model_id}")
        logger.info(f"Model parameters: {model.num_parameters():,}")
        logger.info(f"Tokenizer length: {len(tokenizer)}")
        logger.info(f"Max sequence length: {max_seq_length}")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {str(e)}")
        raise



model, tokenizer = get_model_and_tokenizer(
        model_id=model_id,
        max_seq_length=1000,  # Match with your trainer config
        load_in_8bit=False,   # Use 4-bit by default
        trust_remote_code=True
    )


class ModelInference:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """Initialize the inference class with model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if torch.cuda.is_available() else "cpu"

        # Store the original model state
        self.model_training_mode = self.model.training

        # Default generation config
        self.default_gen_config = GenerationConfig(
            penalty_alpha=0.6,
            do_sample=True,
            top_k=5,
            temperature=0.5,
            repetition_penalty=1.2,
            max_new_tokens=200,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def formatted_prompt(self, question: str) -> str:
        """Format the input prompt with appropriate tokens."""
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

    def generate_response(
        self,
        user_input: str,
        gen_config: Optional[GenerationConfig] = None,
        **kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            prompt = self.formatted_prompt(user_input)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Use a more stable default configuration
            default_gen_config = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

            # Use provided config or default
            gen_config = gen_config or default_gen_config

            # Update config with any provided kwargs
            for key, value in kwargs.items():
                setattr(gen_config, key, value)

            self.model.eval()

            start_time = perf_counter()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference_time = perf_counter() - start_time

            return {
                "response": response_text,
                "inference_time": round(inference_time, 2),
                "input_tokens": inputs.input_ids.shape[-1],
                "output_tokens": outputs.shape[-1]
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

        finally:
            # Restore original model training mode
            self.model.train(self.model_training_mode)


# Example usage:
def test_model(model, tokenizer):
    """Test the model with a sample input."""
    try:
        # Initialize inference class
        inferencer = ModelInference(model, tokenizer)

        # Test input
        test_input = 'Yes, give me an example of code to create a file to create a dataset of size 5x5 and populates it with random values and then outputs it as a csv, in python'

        # Generate response with custom parameters
        result = inferencer.generate_response(
            test_input,
            temperature=0.7,  # Override default temperature
            max_new_tokens=300  # Override default max tokens
        )

        # Print results
        print("\nGenerated Response:")
        print("=" * 50)
        print(result["response"])
        print("\nMetadata:")
        print(f"Inference time: {result['inference_time']} seconds")
        print(f"Input tokens: {result['input_tokens']}")
        print(f"Output tokens: {result['output_tokens']}")

    except Exception as e:
        logger.error(f"Error in test_model: {str(e)}")
        raise


# Test the model
test_model(model, tokenizer)


output_model="llama3.18B-BHK-Lab-Data-Fine-tunedByMogtaba"


import pandas as pd
from datasets import Dataset

def prepare_train_data_v4(data):
    # Initialize an empty list to store each separate prompt-response pair
    formatted_data = []

    # Process each entry in the dataset
    for entry in data:
        # Process question-answer pairs for code packages
        if entry["repo"].startswith("https://github.com"):
          if entry["qa_pairs"]:
            for qa_pair in entry["qa_pairs"]:
                file = entry['file'].split("repos")[-1]
                prompt = (f"Repository: {entry['repo']}\n"
                          f"File Name: {file}\n"
                          f"Language: {entry['language']}\n"
                          f"File Content:\n{entry['content'][:4000]}\n"
                          f"Question: {qa_pair['question']}")
                response = qa_pair['answer']
                formatted_data.append({
                    "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
                })
            # Process completion tasks for code packages
        if "completion_tasks" in entry and entry["completion_tasks"]:
          for completion_task in entry["completion_tasks"]:
            file = entry['file'].split("repos")[-1]
            prompt = (f"Complete the following code:\n{completion_task['partial']}\n"
                      f"Based on the file name: {file}\n"
                      f"With the following content: {entry['content'][:4000]}")
            response = completion_task.get('complete', completion_task['partial'])
            imports = entry["dependencies"]["imports"]
            from_imports = entry["dependencies"]["from_imports"]
            partial_text = "The following is the partial code, provide the completion for this code:\n"
            complete_text = "The following is the complete code:\n"
            formatted_data.append({
                "text": f"<|im_start|>user\n{partial_text}{prompt}<|im_end|>\n<|im_start|>assistant\n {complete_text} Imports: {imports}\n From Imports: {from_imports}\n {response}<|im_end|>\n"
                })

        # Process question-answer pairs for research papers
        elif entry["repo"] == "research_papers":
            for qa_pair in entry["qa_pairs"]:
                prompt = (f"Research Paper: {entry['file']}\n"
                          f"Content Excerpt:\n{entry['content']}\n"
                          f"Question: {qa_pair['question']}")
                response = qa_pair['answer']
                formatted_data.append({
                    "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
                })

    # Convert the list of dictionaries into a DataFrame
    data_df = pd.DataFrame(formatted_data)

    # Create a new Dataset from the DataFrame
    final_dataset = Dataset.from_pandas(data_df)

    return final_dataset


import json
import pandas as pd
from datasets import Dataset

# Load the JSON data
with open('/content/combined_dataset_train.json') as f:
    train_data = json.load(f)

# Load the JSON data
with open('/content/combined_dataset_val.json') as f:
    val_data = json.load(f)



train_data_final = prepare_train_data_v4(train_data)
val_data_final = prepare_train_data_v4(val_data)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
peft_config = LoraConfig(
    r=16,                  # Rank of the update matrices - higher means more capacity to learn but uses more memory
    lora_alpha=32,         # Scaling factor for LoRA updates - higher means stronger influence of fine-tuning
    lora_dropout=0.1,      # Dropout rate for LoRA layers to prevent overfitting
    bias="none",           # Train bias terms using LoRA, giving more flexibility to the model
    task_type="CAUSAL_LM", # Specify this is for causal language modeling (predicting next token)
    # List of model layers to apply LoRA to - these are the attention layers which are most important for adaptation
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Targeting query, value, key, and output projections
)


model.gradient_checkpointing_enable()


# Configure all training-related parameters
training_arguments = TrainingArguments(
    output_dir=output_model,                  # Directory where model checkpoints will be saved
    per_device_train_batch_size=2,           # Number of samples processed per GPU - higher uses more memory
    gradient_accumulation_steps=32,          # Accumulate gradients over 32 steps - effective batch size = 2*32 = 64
    optim="paged_adamw_32bit",              # Use memory-efficient AdamW optimizer with 32-bit precision
    learning_rate=3e-4,                      # Learning rate - controls how big the model updates are
    lr_scheduler_type="cosine_with_restarts", # Learning rate schedule - reduces LR over time with periodic restarts
    warmup_ratio=0.1,                        # Gradually increase LR for first 10% of training to stabilize training
    save_strategy="epoch",                   # Save model checkpoint at the end of each epoch
    logging_steps=10,                        # Log training metrics every 10 steps for monitoring
    num_train_epochs=5,                      # Number of complete passes through the training data
    max_steps=500,                           # Maximum number of training steps, regardless of epochs
    fp16=True,                              # Use 16-bit floating point precision to save memory
    push_to_hub=True,                       # Automatically upload model to Hugging Face Hub
    weight_decay=0.01,                      # L2 regularization to prevent overfitting
    group_by_length=True,                    # Group similar length sequences together for efficiency

    # Add evaluation during training
    evaluation_strategy="epoch",     # Run evaluation at the end of each epoch
    eval_steps=50,                  # Also evaluate every 50 steps

    # Save best model based on evaluation metric
    load_best_model_at_end=True,    # Load the best model at the end of training
    metric_for_best_model="loss",   # Use loss to determine best model

    # Additional logging
    logging_dir="./logs",           # Directory for training logs
    report_to=["tensorboard"],      # Log to tensorboard for visualization
)


torch.cuda.empty_cache()


# Set up the Supervised Fine-Tuning trainer
trainer = SFTTrainer(
    model=model,                            # The base model to fine-tune
    train_dataset=train_data_final,               # The dataset to train on
    peft_config=peft_config,                # LoRA configuration from above
    # dataset_text_field="text",              # Column name in dataset containing the text to train on
    args=training_arguments,                # Training arguments from above
    tokenizer=tokenizer,                    # Tokenizer for converting text to tokens
    # packing=True,                           # Pack multiple sequences together to maximize GPU utilization
    # max_seq_length=1000,                     # Maximum length of input sequences - longer sequences get truncated
    eval_dataset=val_data_final,           # Add validation dataset if available
)

# Before training, you might want to verify the configuration:
print(f"Effective batch size: {training_arguments.per_device_train_batch_size * training_arguments.gradient_accumulation_steps}")
print(f"Number of trainable parameters: {trainer.model.num_parameters(only_trainable=True)}")

torch.cuda.empty_cache()

trainer.train()