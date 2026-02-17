"""
Diagnostic script to run all models and identify crashes
"""
import subprocess
import sys
import time
import torch

print("=" * 70)
print("RUNNING ALL MODELS - CRASH DIAGNOSTICS")
print("=" * 70)

# Check system info
print(f"\n[*] System Information:")
print(f"    Python: {sys.version.split()[0]}")
print(f"    PyTorch: {torch.__version__}")
print(f"    CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

models_to_run = [
    ("train.py", 180, "CIFAR-10 + MNIST training"),
    ("verify_gpu.py", 30, "GPU verification"),
    ("chat_server.py", 60, "Chat server startup"),
]

print("\n" + "=" * 70)
print("RUNNING MODELS")
print("=" * 70)

results = {}

for script, timeout, description in models_to_run:
    print(f"\n[{'=' * 50}]")
    print(f"[{script}] {description}")
    print(f"Timeout: {timeout}s")
    print(f"[{'=' * 50}]")
    
    try:
        start = time.time()
        result = subprocess.run(
            ["python", script],
            capture_output=False,  # Show real-time output
            timeout=timeout,
            check=False
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n[✓] {script} SUCCESS (took {elapsed:.1f}s)")
            results[script] = "SUCCESS"
        else:
            print(f"\n[✗] {script} FAILED with return code {result.returncode}")
            results[script] = f"FAILED (code {result.returncode})"
    
    except subprocess.TimeoutExpired:
        print(f"\n[✗] {script} TIMEOUT after {timeout}s (possible infinite loop)")
        results[script] = f"TIMEOUT ({timeout}s)"
    except Exception as e:
        print(f"\n[✗] {script} ERROR: {e}")
        results[script] = f"ERROR: {e}"
    
    print(f"[{'=' * 50}]\n")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for script, status in results.items():
    print(f"  {script:<25} {status}")
print("=" * 70)
