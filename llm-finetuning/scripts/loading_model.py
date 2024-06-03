import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

def load_tokenizer(model_name):
    """
    Load and return the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_model(model_name, use_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4", use_nested_quant=False):
    """
    Load and return the model with optional QLoRA configuration.
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=bnb_config,
    )
    return model

def configure_lora(r=64, alpha=16, dropout=0.1):
    """
    Configure and return the LoRA settings.
    """
    lora_config = LoraConfig(
        r=r,
        alpha=alpha,
        dropout=dropout
    )
    return lora_config

if __name__ == "__main__":
    from config import model_name, use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant, lora_r, lora_alpha, lora_dropout

    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    print("Tokenizer loaded successfully.")

    # Load model
    model = load_model(
        model_name,
        use_4bit=use_4bit,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        use_nested_quant=use_nested_quant
    )
    print("Model loaded successfully.")

    # Configure LoRA
    lora_config = configure_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    print("LoRA configuration applied successfully.")
