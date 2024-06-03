# Model configuration
model_name = "meta-llama/Llama-2-7b-hf"  # Name of the pre-trained model
checkpoint = "iocuydi/llama-2-amharic-3784m"  # Checkpoint for the model
commit_hash = "04fcac974701f1dab0b8e39af9d3ecfce07b3773"  # Commit hash for the model
OUTPUT_DIR="./results"

# QLoRA parameters
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.1

# bitsandbytes parameters
USE_4BIT=true
BNB_4BIT_COMPUTE_DTYPE="float16"
BNB_4BIT_QUANT_TYPE="nf4"
USE_NESTED_QUANT=false

# TrainingArguments parameters
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
GRADIENT_CHECKPOINTING=true
MAX_GRAD_NORM=0.3
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.001
OPTIM="paged_adamw_32bit"
LR_SCHEDULER_TYPE="cosine"
MAX_STEPS=-1
WARMUP_RATIO=0.03
GROUP_BY_LENGTH=true
SAVE_STEPS=0
LOGGING_STEPS=25

# Paths
EXPANSION_FILE_DIR="/content/llm-finetunning/short_forms.txt"
BIGRAM_DIR="bigrams.txt"

# Run the training script with the specified parameters
python finetuning.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --use_4bit $USE_4BIT \
    --bnb_4bit_compute_dtype $BNB_4BIT_COMPUTE_DTYPE \
    --bnb_4bit_quant_type $BNB_4BIT_QUANT_TYPE \
    --use_nested_quant $USE_NESTED_QUANT \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --max_grad_norm $MAX_GRAD_NORM \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --optim $OPTIM \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --max_steps $MAX_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --group_by_length $GROUP_BY_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --expansion_file_dir $EXPANSION_FILE_DIR \
    --bigram_dir $BIGRAM_DIR
