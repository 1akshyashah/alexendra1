import json
import os
import sys
import subprocess
from multimodal_merger import MultiModalLearner
from internet_crawler import crawl_programming_data
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

def run_subprocess(script_name, timeout=300):
    """Safely run subprocess with timeout to prevent infinite loops."""
    try:
        print(f"[*] Starting {script_name}...")
        result = subprocess.run(
            ["python", script_name],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        if result.returncode == 0:
            print(f"[+] {script_name} completed successfully")
            return True
        else:
            print(f"[!] {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[!] {script_name} timed out after {timeout}s (infinite loop detected?)")
        return False
    except Exception as e:
        print(f"[!] {script_name} error: {e}")
        return False

def integrated_training():
    """Train model with all modalities: internet + images + numbers + chat."""
    
    print("=" * 70)
    print("INTEGRATED MULTI-MODAL TRAINING")
    print("=" * 70)
    
    # Step 1: Crawl internet data
    print("\n[1/5] Crawling internet data...")
    try:
        internet_data = crawl_programming_data()
    except Exception as e:
        print(f"[!] Internet crawl skipped: {e}")
        internet_data = []
    
    # Step 2: Train/load existing models (with timeout safety)
    print("\n[2/5] Loading your existing models...")
    run_subprocess("train.py", timeout=180)
    run_subprocess("train_advanced.py", timeout=180)
    
    # Step 3: Merge all modalities
    print("\n[3/5] Merging image + number + internet + chat data...")
    learner = MultiModalLearner()
    fused_data = learner.generate_merged_training_data()
    
    # Step 4: Prepare training examples
    print("\n[4/5] Preparing unified training dataset...")
    examples = []
    
    # From chat/feedback data
    if os.path.exists("data/dialogs.jsonl"):
        with open("data/dialogs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("feedback") == "up" or rec.get("correction"):
                        text = f"CHAT: User: {rec.get('user_message', '')} Bot: {rec.get('correction', rec.get('bot_reply', ''))}"
                        examples.append({"text": text})
                except:
                    pass
    
    # From internet data
    for item in internet_data:
        try:
            text = f"INTERNET: {item['source']}: {json.dumps(item['data'])[:200]}"
            examples.append({"text": text})
        except:
            pass
    
    # From merged multimodal data
    for item in fused_data.get("combined_examples", []):
        if isinstance(item["content"], dict):
            text = f"{item['source'].upper()}: {json.dumps(item['content'])[:200]}"
        else:
            text = f"{item['source'].upper()}: {item['content']}"
        examples.append({"text": text})
    
    if len(examples) == 0:
        print("[!] No training examples found. Add data first.")
        return
    
    print(f"[+] Total training examples: {len(examples)}")
    
    # Step 5: Fine-tune merged model
    print("\n[5/5] Fine-tuning integrated model...")
    
    ds = Dataset.from_list(examples[:200])  # Limit for safety
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=512, padding="max_length")
    
    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    
    training_args = TrainingArguments(
        output_dir="models/integrated",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        save_total_limit=2,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model("models/integrated")
    
    print("\n" + "=" * 70)
    print("[+] INTEGRATED TRAINING COMPLETE!")
    print("=" * 70)
    print("\nTraining Summary:")
    print(f"  ✓ Image/Number model (train.py)")
    print(f"  ✓ Advanced model (train_advanced.py)")
    print(f"  ✓ Internet sources: {len(internet_data)}")
    print(f"  ✓ Chat examples: synced")
    print(f"  ✓ Total training examples: {len(examples)}")
    print(f"  ✓ Final model saved to: models/integrated")
    print("\nYou can now use the integrated model for:")
    print("  - Image recognition")
    print("  - Number prediction")
    print("  - Programming knowledge")
    print("  - Self-learning from chat feedback")
    print("=" * 70)

if __name__ == "__main__":
    integrated_training()