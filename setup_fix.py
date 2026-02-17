#!/usr/bin/env python
"""Setup and fix script to resolve crashes."""
import os
import json

print("=" * 70)
print("FIXING PROJECT SETUP")
print("=" * 70)

# 1. Create missing directories
print("\n[1/3] Creating missing data directories...")
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/internet_data", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("[+] Directories created")

# 2. Create dummy data files if they don't exist
print("\n[2/3] Creating placeholder data files...")

if not os.path.exists("data/numbers.json"):
    with open("data/numbers.json", "w", encoding="utf-8") as f:
        json.dump([{"number": 1}, {"number": 2}], f)
    print("[+] Created data/numbers.json")

if not os.path.exists("data/internet_data/crawl_latest.jsonl"):
    with open("data/internet_data/crawl_latest.jsonl", "w", encoding="utf-8") as f:
        f.write('{"source": "placeholder", "data": "dummy data"}\n')
    print("[+] Created data/internet_data/crawl_latest.jsonl")

# 3. Verify fixes
print("\n[3/3] Verifying project structure...")
required_dirs = ["data/images", "data/internet_data", "models"]
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"[✓] {dir_path}")
    else:
        print(f"[✗] {dir_path} FAILED")

required_files = ["data/numbers.json", "data/internet_data/crawl_latest.jsonl"]
for file_path in required_files:
    if os.path.exists(file_path):
        print(f"[✓] {file_path}")
    else:
        print(f"[✗] {file_path} FAILED")

print("\n" + "=" * 70)
print("SETUP COMPLETE - Ready to run models")
print("=" * 70)
