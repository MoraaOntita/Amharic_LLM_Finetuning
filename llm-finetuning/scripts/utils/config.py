# config.py

# Model configuration
model_name = "meta-llama/Llama-2-7b-hf"  # Name of the pre-trained model
checkpoint = "iocuydi/llama-2-amharic-3784m"  # Checkpoint for the model
commit_hash = "04fcac974701f1dab0b8e39af9d3ecfce07b3773"  # Commit hash for the model


# QLoRA parameters
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# bitsandbytes parameters
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# TrainingArguments parameters
OUTPUT_DIR = "./results"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = True
MAX_GRAD_NORM = 0.3
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
MAX_STEPS = -1
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True
SAVE_STEPS = 0
LOGGING_STEPS = 25

# Data paths
EXPANSION_FILE_DIR = '/content/llm-finetunning/short_forms.txt'
BIGRAM_DIR = 'bigrams.txt'

# Dataset configuration
DATASET_NAMES = [
    "rasyosef/amharic-news-category-classification",
    "simonbutt/amharic_truthful_qa",
    "simonbutt/amharic_gsm8k",
    "EthioNLP/Amharic_LLAMA_MT",
    "EthioNLP/Amharic_Instruction_dataset",
    "Tvsybkzkmapab/Amharic_ad_generation",
    "BiniyamAjaw/amharic_dataset_v2",
    "Henok/amharic-qa"
]

TEXT_FIELDS = ['headline', 'article', 'question', 'am_question', 'am_answer', 'instruction', 'input', 'output', 'text', 'targets', 'inputs']
