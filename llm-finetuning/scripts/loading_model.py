import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from utils.config import model_name, checkpoint, commit_hash, use_4bit, bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant, lora_r, lora_alpha, lora_dropout
from transformers import PeftModel


def load_tokenizer():
    """
    Load and return the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_model():
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
        checkpoint,
        config=bnb_config,
        revision=commit_hash
    )
    return model

def resize_token_embeddings(tokenizer, model):
    """
    Resize the token embeddings of the model to match the tokenizer's vocabulary size.
    """
    model.resize_token_embeddings(len(tokenizer))
    return model

def load_peft_model(llama_model, pretrained_model_name_or_path, revision, cache_dir=None):
    """
    Load and return the Peft model.

    Args:
        llama_model (str): Path or name of the Llama model used as base.
        pretrained_model_name_or_path (str): Name or path of the pre-trained Peft model.
        revision (str): Commit hash or tag of the model to load.
        cache_dir (str, optional): Directory for caching downloaded model files.

    Returns:
        PeftModel: Loaded Peft model.
    """
    model = PeftModel.from_pretrained(
        llama_model,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        cache_dir=cache_dir
    )
    return model

def main():
    # Load tokenizer
    tokenizer = load_tokenizer()
    print("Tokenizer loaded successfully.")

    # Load model
    model = load_model()
    print("Model loaded successfully.")

    # Resize token embeddings
    model = resize_token_embeddings(tokenizer, model)
    print("Token embeddings resized successfully.")

    # Load Peft model
    peft_model = load_peft_model(tokenizer)
    print("Peft model loaded successfully.")

if __name__ == "__main__":
    main()


