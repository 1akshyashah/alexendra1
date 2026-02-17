import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

LOG_FILE = "data/dialogs.jsonl"
OUTPUT_DIR = "models/retrained"
BASE_MODEL = "distilgpt2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_gpu():
    """Configure GPU for training."""
    if not torch.cuda.is_available():
        print("[!] GPU not available! Will use CPU for training.")
        return False
    
    print(f"[+] CUDA Available: {torch.cuda.is_available()}")
    print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    return True

gpu_available = setup_gpu()

def load_interactions(log_file: str) -> list:
    """Load all interactions from JSONL log."""
    interactions = []
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return interactions
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                interactions.append(rec)
            except json.JSONDecodeError:
                continue
    return interactions

def build_training_examples(interactions: list) -> list:
    """Convert interactions to (prompt, target) pairs for fine-tuning."""
    examples = []
    
    for rec in interactions:
        user_msg = rec.get("user_message", "")
        bot_reply = rec.get("bot_reply", "")
        feedback = rec.get("feedback")
        correction = rec.get("correction")
        
        if correction:
            target = correction
            examples.append({
                "text": f"### User:\n{user_msg}\n\n### Assistant:\n{target}\n\n"
            })
        elif feedback == "up":
            examples.append({
                "text": f"### User:\n{user_msg}\n\n### Assistant:\n{bot_reply}\n\n"
            })
    
    return examples

def retrain_model():
    """Fine-tune model on GPU with optimized settings."""
    print("[*] Loading interactions...")
    interactions = load_interactions(LOG_FILE)
    print(f"[*] Found {len(interactions)} total interactions")
    
    print("[*] Building training examples...")
    examples = build_training_examples(interactions)
    print(f"[*] Created {len(examples)} training examples")
    
    if len(examples) == 0:
        print("[!] No training examples. Submit feedback via chat first.")
        return
    
    ds = Dataset.from_list(examples)
    
    print("[*] Loading base model & tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if gpu_available else torch.float32,
        device_map="auto"
    )
    
    def tokenize_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
    
    print("[*] Tokenizing dataset...")
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, batched=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8 if gpu_available else 2,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=10,
        save_steps=50,
        eval_strategy="no",
        seed=42,
        fp16=gpu_available,
        dataloader_pin_memory=gpu_available,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )
    
    print("[*] Starting GPU-accelerated fine-tuning...")
    trainer.train()
    
    print(f"[+] Model saved to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    if gpu_available:
        torch.cuda.empty_cache()
    print("[+] Retrain complete!")

if __name__ == "__main__":
    retrain_model()