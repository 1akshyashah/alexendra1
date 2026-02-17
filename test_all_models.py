"""
Test script to run all models and check for crashes.
"""
import subprocess
import sys
import time
import os
import json

def ensure_data_structure():
    """Ensure all required data directories and files exist."""
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/internet_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create placeholder files if they don't exist
    if not os.path.exists("data/numbers.json"):
        with open("data/numbers.json", "w", encoding="utf-8") as f:
            json.dump([{"number": i} for i in range(1, 6)], f)
    
    if not os.path.exists("data/internet_data/crawl_latest.jsonl"):
        with open("data/internet_data/crawl_latest.jsonl", "w", encoding="utf-8") as f:
            f.write('{"type": "code", "source": "placeholder"}\n')
    
    if not os.path.exists("data/dialogs.jsonl"):
        with open("data/dialogs.jsonl", "w", encoding="utf-8") as f:
            f.write('{"user_message": "hello", "bot_reply": "hi", "feedback": "up"}\n')

def run_test(script_name, timeout=120, description=""):
    """Run a test script with timeout."""
    print(f"\n{'='*70}")
    print(f"Testing: {script_name}")
    if description:
        print(f"Description: {description}")
    print(f"Timeout: {timeout}s")
    print('='*70)
    
    try:
        start = time.time()
        result = subprocess.run(
            [sys.executable, script_name],
            timeout=timeout,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"\n✓ {script_name} SUCCESS ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n✗ {script_name} FAILED (code {result.returncode})")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"\n✗ {script_name} TIMEOUT ({timeout}s) - Possible infinite loop!")
        return False
    except Exception as e:
        print(f"\n✗ {script_name} ERROR: {e}")
        return False

if __name__ == "__main__":
    print("="*70)
    print("ALEXEDRA PROJECT - CRASH DIAGNOSTICS")
    print("="*70)
    
    # Step 1: Setup
    print("\n[*] Step 1: Ensuring data structure...")
    ensure_data_structure()
    print("[✓] Data structure ready")
    
    # Step 2: Run models
    print("\n[*] Step 2: Running models...\n")
    
    results = {}
    
    # Simple test: verify GPU
    results["verify_gpu.py"] = run_test(
        "verify_gpu.py", 
        timeout=30,
        description="Check GPU availability"
    )
    
    # Train models
    results["train_advanced.py"] = run_test(
        "train_advanced.py",
        timeout=300,
        description="Advanced model training"
    )
    
    results["train.py"] = run_test(
        "train.py",
        timeout=300,
        description="Standard training"
    )
    
    # Integrated training
    results["integrated_train.py"] = run_test(
        "integrated_train.py",
        timeout=300,
        description="Multi-modal integrated training"
    )
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for script, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {script:<30} {status}")
    
    print(f"\nResults: {passed}/{total} passed")
    print("="*70)
    
    sys.exit(0 if passed == total else 1)
