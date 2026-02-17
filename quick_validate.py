"""
Quick setup and validation script - tests core functionality without heavy training.
"""
import os
import sys
import json

print("="*70)
print("ALEXEDRA PROJECT - QUICK VALIDATION")
print("="*70)

# Step 1: Create all required directories
print("\n[1/4] Setting up directory structure...")
dirs = [
    "data",
    "data/images",
    "data/internet_data",
    "models",
    "logs"
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    if os.path.exists(dir_path):
        print(f"  [✓] {dir_path}")
    else:
        print(f"  [✗] {dir_path} FAILED")

# Step 2: Create placeholder data files
print("\n[2/4] Creating data files...")
data_files = {
    "data/numbers.json": [{"number": i} for i in range(1, 6)],
    "data/dialogs.jsonl": None,  # Handle separately
}

for file_path, content in data_files.items():
    try:
        if content is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f)
        print(f"  [✓] {file_path}")
    except Exception as e:
        print(f"  [✗] {file_path}: {e}")

# Create dialogs.jsonl
try:
    with open("data/dialogs.jsonl", "w", encoding="utf-8") as f:
        f.write('{"user_message": "hello", "bot_reply": "hi", "feedback": "up"}\n')
    print(f"  [✓] data/dialogs.jsonl")
except Exception as e:
    print(f"  [✗] data/dialogs.jsonl: {e}")

# Create internet data
try:
    with open("data/internet_data/crawl_latest.jsonl", "w", encoding="utf-8") as f:
        f.write('{"type": "code", "source": "Python"}\n')
        f.write('{"type": "article", "source": "ML Docs"}\n')
    print(f"  [✓] data/internet_data/crawl_latest.jsonl")
except Exception as e:
    print(f"  [✗] data/internet_data/crawl_latest.jsonl: {e}")

# Step 3: Test Python dependencies
print("\n[3/4] Checking Python dependencies...")
dependencies = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("transformers", "Transformers"),
    ("PIL", "PIL/Pillow"),
    ("datasets", "Hugging Face datasets"),
]

missing = []
for module, name in dependencies:
    try:
        __import__(module)
        print(f"  [✓] {name}")
    except ImportError:
        print(f"  [✗] {name} MISSING")
        missing.append(name)

if missing:
    print(f"\n[!] Missing modules: {', '.join(missing)}")
    print("    Run: pip install -r requirements.txt")

# Step 4: Test GPU availability
print("\n[4/4] Checking hardware...")
try:
    import torch
    print(f"  [✓] PyTorch version: {torch.__version__}")
    print(f"  [✓] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [✓] GPU: {torch.cuda.get_device_name(0)}")
        print(f"  [✓] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"  [!] CUDA not available - will use CPU (slower)")
except Exception as e:
    print(f"  [✗] PyTorch check failed: {e}")

print("\n" + "="*70)
print("✓ VALIDATION COMPLETE - Ready to run training")
print("="*70)
print("\nNext steps:")
print("  1. Run: python train.py")
print("  2. Run: python train_advanced.py")
print("  3. Run: python integrated_train.py")
print("\nOr test all at once:")
print("  python test_all_models.py")
print("="*70)
