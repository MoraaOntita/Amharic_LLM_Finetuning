import argparse
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
from utils.common import preprocess_dataset
from utils.config import TEXT_FIELDS

def main(args):
    # Load the tokenizer and model with QLoRA configuration
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

    # Set LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )

    # Set Training Arguments
    training_args = TrainingArguments(
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

    # Initialize SFTTrainer
    preprocessor = AmharicPreprocessor(args.expansion_file_dir, args.bigram_dir)
    datasets = load_and_preprocess_datasets(preprocessor)
    train_dataset = datasets['train']
    eval_dataset = datasets['eval']

    trainer = SFTTrainer(
        model=llama_model,
        tokenizer=llama_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and dataset configuration
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--output_dir", type=str, default="./results")

    # QLoRA parameters
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # bitsandbytes parameters
    parser.add_argument("--use_4bit", type=bool, default=True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--use_nested_quant", type=bool, default=False)

    # TrainingArguments parameters
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=25)

    # Data paths
    parser.add_argument("--expansion_file_dir", type=str, default="/content/llm-finetunning/short_forms.txt")
    parser.add_argument("--bigram_dir", type=str, default="bigrams.txt")

    args = parser.parse_args()

    main(args)
