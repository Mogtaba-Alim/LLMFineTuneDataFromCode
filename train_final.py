#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Llama 3.1 8B model on code and research paper data.
This script handles diverse data formats and optimizes for high-quality fine-tuning
with an 8192 token context window.
"""

import os
import json
import logging
import random
import argparse
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
def set_seed(seed_value: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    logger.info(f"Random seed set to {seed_value}")


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_rank: int = field(
        default=16,
        metadata={"help": "Rank for LoRA adaptation"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "Alpha parameter for LoRA"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_nested_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use nested quantization for 4-bit (double quantization)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit quantization (fp4 or nf4)"}
    )
    use_flash_attn: bool = field(
        default=True, 
        metadata={"help": "Whether to use flash attention for faster training"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_file: str = field(
        default="final_combined_train_dataset.json",
        metadata={"help": "Path to training data file"}
    )
    val_file: str = field(
        default="final_combined_val_dataset.json",
        metadata={"help": "Path to validation data file"}
    )
    max_seq_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length for the model"}
    )
    add_special_tokens: bool = field(
        default=True, 
        metadata={"help": "Whether to add special tokens to the data"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for training configuration."""
    output_dir: str = field(
        default="./llama-3.1-8b-finetuned",
        metadata={"help": "Output directory for model and checkpoints"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of updates steps to accumulate before backward pass"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Initial learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay applied to parameters"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps"}
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule type"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 mixed precision training"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Whether to use bf16 mixed precision training"}
    )
    optim: str = field(
        default="paged_adamw_8bit",
        metadata={"help": "Optimizer to use for training"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    eval_steps: int = field(
        default=100,
        metadata={"help": "Run evaluation every X steps"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints to save space"}
    )
    report_to: str = field(
        default="tensorboard",
        metadata={"help": "The integration to report the results and logs to"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )


class CustomDataset(Dataset):
    """Custom dataset for handling code and research paper data in various formats."""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"Loading data from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Successfully loaded {len(data)} items from {file_path}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
        
        skipped_items = 0
        logger.info(f"Processing {len(data)} examples")
        
        for idx, item in enumerate(data):
            try:
                examples_before = len(self.examples)
                self._process_item(item)
                examples_added = len(self.examples) - examples_before
                
                if examples_added == 0:
                    skipped_items += 1
                    if skipped_items <= 5:  # Only log the first few skipped items to avoid flooding logs
                        logger.warning(f"Item {idx} did not generate any examples. Keys: {list(item.keys())}")
            except Exception as e:
                logger.error(f"Error processing item {idx}: {str(e)}")
                # Continue processing other items
        
        logger.info(f"Created {len(self.examples)} training examples from {len(data)} items. Skipped {skipped_items} items.")
    
    def _process_item(self, item: Dict[str, Any]) -> None:
        """Process a single data item into training examples."""
        
        # Determine data type based on format
        is_code_data = "language" in item and item["language"] not in ["research_paper"]
        
        # Process QA pairs
        if item and "qa_pairs" in item and item["qa_pairs"] is not None:
            qa_examples = []
            for qa_pair in item["qa_pairs"]:
                # Check for required keys
                if "question" not in qa_pair or "answer" not in qa_pair:
                    continue
                
                # Different formats for research papers
                if not is_code_data and "chunk" in qa_pair:
                    # Research paper format 1
                    context = qa_pair.get("chunk", "")
                    prompt = f"Context:\n{context}\n\nQuestion: {qa_pair['question']}\n\nAnswer:"
                    response = qa_pair["answer"]
                else:
                    # Code format or Research paper format 2
                    content = item.get("content", "")
                    
                    # For code data, use file content as context
                    if is_code_data:
                        if len(content) > 0:
                            prompt = f"File content:\n```{item.get('language', '')}\n{content}\n```\n\nQuestion: {qa_pair['question']}\n\nAnswer:"
                        else:
                            prompt = f"Question: {qa_pair['question']}\n\nAnswer:"
                    else:
                        # For research papers without specific chunks
                        prompt = f"Question about research paper '{item.get('file', '')}': {qa_pair['question']}\n\nAnswer:"
                    
                    response = qa_pair["answer"]
                
                qa_examples.append({"prompt": prompt, "response": response})
            
            self.examples.extend(qa_examples)
        
        # Process code completion tasks
        if is_code_data and "completion_tasks" in item and item["completion_tasks"] is not None:
            for task in item["completion_tasks"]:
                if "partial" in task and ("complete" in task or "completion" in task):
                    prompt = f"Complete the following code:\n```{item.get('language', '')}\n{task['partial']}\n```"
                    # Some datasets might use 'completion' instead of 'complete'
                    response = task.get("complete", task.get("completion", ""))
                    self.examples.append({"prompt": prompt, "response": response})
        
        # Process debugging tasks
        if is_code_data and "debugging_tasks" in item and item["debugging_tasks"] is not None:
            for task in item["debugging_tasks"]:
                if "bug_description" in task and "bug_fix" in task:
                    prompt = f"Fix the following bug in the code:\nBug description: {task['bug_description']}\n```{item.get('language', '')}\n{item.get('content', '')}\n```"
                    response = task["bug_fix"]
                    self.examples.append({"prompt": prompt, "response": response})
        
        # Process refactoring tasks
        if is_code_data and "refactoring_tasks" in item and item["refactoring_tasks"] is not None:
            for task in item["refactoring_tasks"]:
                if "original_snippet" in task and "refactored_snippet" in task:
                    prompt = f"Refactor the following code:\n```{item.get('language', '')}\n{task['original_snippet']}\n```"
                    explanation = task.get("explanation", "")
                    response = f"```{item.get('language', '')}\n{task['refactored_snippet']}\n```"
                    if explanation:
                        response += f"\n\nExplanation: {explanation}"
                    self.examples.append({"prompt": prompt, "response": response})
        
        # Process docstring tasks
        if is_code_data and "docstring_tasks" in item and item["docstring_tasks"] is not None:
            for task in item["docstring_tasks"]:
                if "function_signature" in task and "docstring" in task:
                    prompt = f"Write a docstring for the following function:\n```{item.get('language', '')}\n{task['function_signature']}\n```"
                    response = task["docstring"]
                    self.examples.append({"prompt": prompt, "response": response})
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        prompt = example["prompt"]
        response = example["response"]
        
        # Format for Llama 3.1 chat format
        formatted_prompt = f"<|begin_of_text|><|user|>\n{prompt}<|end_of_turn|>\n<|assistant|>\n"
        formatted_response = f"{response}<|end_of_turn|>"
        
        full_text = formatted_prompt + formatted_response
        
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels - we only want to compute loss on the assistant's response
        input_ids = encoded["input_ids"][0]
        labels = input_ids.clone()
        
        # Mask out the prompt tokens for loss calculation
        prompt_encoded = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        labels[:len(prompt_encoded)] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoded["attention_mask"][0],
            "labels": labels,
        }


def format_for_trl(item: Dict[str, Any]) -> Dict[str, str]:
    """Format an example for TRL's SFTTrainer."""
    return {
        "prompt": item["prompt"],
        "completion": item["response"],
    }


def prepare_dataset_for_trl(file_path: str) -> List[Dict[str, str]]:
    """Prepare dataset in the format expected by TRL's SFTTrainer."""
    logger.info(f"Loading data from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Successfully loaded {len(data)} items from {file_path}")
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise
    
    formatted_data = []
    skipped_items = 0
    
    for idx, item in enumerate(data):
        try:
            # Determine data type based on format
            is_code_data = "language" in item and item["language"] not in ["research_paper"]
            
            examples_added = 0
            
            # Process QA pairs
            if item and "qa_pairs" in item and item["qa_pairs"] is not None:
                for qa_pair in item["qa_pairs"]:
                    # Different formats for research papers
                    if not is_code_data and "chunk" in qa_pair:
                        # Research paper format 1
                        context = qa_pair.get("chunk", "")
                        prompt = f"Context:\n{context}\n\nQuestion: {qa_pair['question']}\n\nAnswer:"
                        response = qa_pair["answer"]
                    else:
                        # Code format or Research paper format 2
                        content = item.get("content", "")
                        
                        # For code data, use file content as context
                        if is_code_data:
                            if len(content) > 0:
                                prompt = f"File content:\n```{item.get('language', '')}\n{content}\n```\n\nQuestion: {qa_pair['question']}\n\nAnswer:"
                            else:
                                prompt = f"Question: {qa_pair['question']}\n\nAnswer:"
                        else:
                            # For research papers without specific chunks
                            prompt = f"Question about research paper '{item.get('file', '')}': {qa_pair['question']}\n\nAnswer:"
                        
                        response = qa_pair["answer"]
                    
                    formatted_data.append({"prompt": prompt, "completion": response})
                    examples_added += 1
            
            # Process code completion tasks
            if is_code_data and "completion_tasks" in item and item["completion_tasks"] is not None:
                for task in item["completion_tasks"]:
                    if "partial" in task and ("complete" in task or "completion" in task):
                        prompt = f"Complete the following code:\n```{item.get('language', '')}\n{task['partial']}\n```"
                        # Some datasets might use 'completion' instead of 'complete'
                        response = task.get("complete", task.get("completion", ""))
                        formatted_data.append({"prompt": prompt, "completion": response})
                        examples_added += 1
            
            # Process debugging tasks
            if is_code_data and "debugging_tasks" in item and item["debugging_tasks"] is not None:
                for task in item["debugging_tasks"]:
                    if "bug_description" in task and "bug_fix" in task:
                        prompt = f"Fix the following bug in the code:\nBug description: {task['bug_description']}\n```{item.get('language', '')}\n{item.get('content', '')}\n```"
                        response = task["bug_fix"]
                        formatted_data.append({"prompt": prompt, "completion": response})
                        examples_added += 1
            
            # Process refactoring tasks
            if is_code_data and "refactoring_tasks" in item and item["refactoring_tasks"] is not None:
                for task in item["refactoring_tasks"]:
                    if "original_snippet" in task and "refactored_snippet" in task:
                        prompt = f"Refactor the following code:\n```{item.get('language', '')}\n{task['original_snippet']}\n```"
                        explanation = task.get("explanation", "")
                        response = f"```{item.get('language', '')}\n{task['refactored_snippet']}\n```"
                        if explanation:
                            response += f"\n\nExplanation: {explanation}"
                        formatted_data.append({"prompt": prompt, "completion": response})
                        examples_added += 1
            
            # Process docstring tasks
            if is_code_data and "docstring_tasks" in item and item["docstring_tasks"] is not None:
                for task in item["docstring_tasks"]:
                    if "function_signature" in task and "docstring" in task:
                        prompt = f"Write a docstring for the following function:\n```{item.get('language', '')}\n{task['function_signature']}\n```"
                        response = task["docstring"]
                        formatted_data.append({"prompt": prompt, "completion": response})
                        examples_added += 1
            
            if examples_added == 0:
                skipped_items += 1
                if skipped_items <= 5:  # Only log the first few skipped items to avoid flooding logs
                    logger.warning(f"Item {idx} did not generate any examples. Keys: {list(item.keys())}")
        
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            # Continue processing other items
    
    logger.info(f"Created {len(formatted_data)} examples from {len(data)} items. Skipped {skipped_items} items.")
    
    return formatted_data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.1 on code and research paper data")
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set random seed for reproducibility
    set_seed(training_args.seed)
    
    # Set up 4-bit quantization configuration if enabled
    compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
    
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )
    
    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        use_cache=False,  # Gradient checkpointing requires this
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if model_args.use_flash_attn else "eager",
        device_map="auto",
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=True,
    )
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training if using LoRA
    if model_args.use_lora:
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    logger.info("Model and tokenizer loaded successfully")
    
    # Prepare the datasets for training
    logger.info(f"Processing training data from: {data_args.train_file}")
    train_dataset_list = prepare_dataset_for_trl(data_args.train_file)
    train_dataset = Dataset.from_list(train_dataset_list)

    if data_args.val_file:
        logger.info(f"Processing validation data from: {data_args.val_file}")
        eval_dataset_list = prepare_dataset_for_trl(data_args.val_file)
        eval_dataset = Dataset.from_list(eval_dataset_list)
    else:
        eval_dataset = None
    
    # Create the SFT Trainer
    trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        num_train_epochs=training_args.num_train_epochs,
        warmup_ratio=training_args.warmup_ratio,
        lr_scheduler_type=training_args.lr_scheduler_type,
        gradient_checkpointing=training_args.gradient_checkpointing,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        optim=training_args.optim,
        logging_steps=training_args.logging_steps,
        eval_steps=training_args.eval_steps,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        report_to=training_args.report_to,
        seed=training_args.seed,
        max_seq_length=data_args.max_seq_length,
        dataset_num_proc=1,  # Adjust as needed
        packing=False,
        dataset_text_field="text",  # Include this if your text field is not the default "text"
    ),
)

    
    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    # Upload the model and tokenizer to the Hugging Face Hub
    logger.info("Uploading model to Hugging Face Hub")
    model.push_to_hub("llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba")
    tokenizer.push_to_hub("llama3.1-8B-BHK-LABGPT-Fine-tunedByMogtaba")
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()