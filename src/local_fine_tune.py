import torch
import sys
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Force flush of stdout to see prints immediately
sys.stdout.reconfigure(line_buffering=True)

# --- Diagnostic Checks ---
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    # Clear cache to free up as much VRAM as possible
    torch.cuda.empty_cache()
else:
    print("WARNING: CUDA is NOT available. Training will likely fail or be extremely slow.")

# CHANGED: Switched to 1.5B model to fit in RTX 3050 4GB VRAM without quantization
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 1024 # Reduced slightly to ensure memory safety on 4GB card

TRAIN_SET_PATH = 'C:/Users/brian/Documents/Random/scibowl-gpt/data/train/mit_ess.jsonl'
OUTPUT_DIR = "C:/Users/brian/Documents/Random/scibowl-gpt/models/qwen2.5-1.5b-ess-scibowl-11182025"

print(f"Loading model: {MODEL_NAME} in float16...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,       
        device_map={"": 0},        
        trust_remote_code=False, # CHANGED: Use internal stable implementation
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR Loading Model: {e}")
    raise e

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=False, 
)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

print("Applying LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",      
        "gate_proj", "up_proj", "down_proj"          
    ]
)

print(f"Loading Dataset: {TRAIN_SET_PATH}")
dataset = load_dataset("json", data_files=TRAIN_SET_PATH)["train"]

def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

dataset = dataset.map(
    lambda x: {"text": format_chat(x)},
    remove_columns=dataset.column_names,
    desc="Formatting chat data"
)

print("Initializing Trainer...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # CHANGED: Reduced to 1 for 4GB VRAM safety
    gradient_accumulation_steps=16, # Increased to compensate for low batch size
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=5,
    optim="adamw_torch", 
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    fp16=True,  
    bf16=False,
    gradient_checkpointing=True, # ADDED: Critical for saving VRAM
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config, 
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=training_args,
    packing=False, 
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")