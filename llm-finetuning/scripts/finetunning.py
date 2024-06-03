import argparse
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig
from loading_dataset import load_and_preprocess_datasets
from utils.common import preprocess_dataset, compute_metrics
from utils.config import TEXT_FIELDS
from preprocessing import AmharicPreprocessor
from typing import Any, Dict

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exception_handler(func):
    """Decorator to handle exceptions in functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@exception_handler
def load_model_and_tokenizer(args: argparse.Namespace) -> (AutoModelForCausalLM, AutoTokenizer):
    """Load the model and tokenizer with QLoRA configuration."""
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )
    
    llama_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=bnb_config,
    )
    
    return llama_model, llama_tokenizer

@exception_handler
def set_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Set training arguments."""
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        group_by_length=args.group_by_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

@exception_handler
def main(args: argparse.Namespace) -> None:
    """Main function to load datasets, model, tokenizer and start training."""
    logger.info("Loading model and tokenizer...")
    llama_model, llama_tokenizer = load_model_and_tokenizer(args)

    logger.info("Setting LoRA configuration...")
    lora_config = LoraConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    logger.info("Setting training arguments...")
    training_args = set_training_args(args)

    logger.info("Loading and preprocessing datasets...")
    preprocessor = AmharicPreprocessor(args.expansion_file_dir, args.bigram_dir)
    datasets = load_and_preprocess_datasets(preprocessor)
    train_dataset = datasets['train']
    eval_dataset = datasets['eval']

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=llama_model,
        tokenizer=llama_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning script for Llama-2 model with QLoRA configuration.")

    # Model and dataset configuration
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf", help="Name of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model checkpoints and predictions")

    # QLoRA parameters
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout probability for LoRA layers")

    # bitsandbytes parameters
    parser.add_argument("--use_4bit", type=bool, default=True, help="Activate 4-bit precision base model loading")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", help="Compute dtype for 4-bit base models")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    parser.add_argument("--use_nested_quant", type=bool, default=False, help="Activate nested quantization for 4-bit base models")

    # TrainingArguments parameters
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per GPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate the gradients for")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Enable gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Maximum gradient normal (gradient clipping)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate (AdamW optimizer)")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay to apply to all layers except bias/LayerNorm weights")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit", help="Optimizer to use")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate schedule")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of training steps (overrides num_train_epochs)")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of steps for a linear warmup (from 0 to learning rate)")
    parser.add_argument("--group_by_length", type=bool, default=True, help="Group sequences into batches with same length")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps")
    parser.add_argument("--logging_steps", type=int, default=25, help="Log every X updates steps")

    # Data paths
    parser.add_argument("--expansion_file_dir", type=str, default="/content/llm-finetunning/short_forms.txt", help="Path to short forms file")
    parser.add_argument("--bigram_dir", type=str, default="bigrams.txt", help="Path to bigram file")

    args = parser.parse_args()

    main(args)
