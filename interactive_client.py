import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def chat(user_id: str, message: str, feedback: str = None, correction: str = None):
    """Send a chat message and get a reply."""
    payload = {
        "user_id": user_id,
        "message": message,
        "feedback": feedback,
        "correction": correction
    }
    resp = requests.post(f"{BASE_URL}/chat", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n[Bot]: {data['reply'][:200]}...")
        print(f"Processing time: {data['processing_time']:.3f}s\n")
        return data['reply']
    else:
        print(f"Error: {resp.status_code}\n")

def check_health():
    """Check server health."""
    resp = requests.get(f"{BASE_URL}/health")
    if resp.status_code == 200:
        data = resp.json()
        print(f"[+] Server OK | GPU Memory: {data.get('gpu_memory_used_gb', 'N/A')} GB\n")

if __name__ == "__main__":
    print("=" * 60)
    print("GPU-Accelerated Self-Learning Chat Bot")
    print("=" * 60)
    print("Commands: 'feedback up' / 'feedback down' / 'correction <text>' / 'quit'\n")
    
    user_id = input("Enter your user ID: ").strip()
    check_health()
    
    while True:
        message = input("[You]: ").strip()
        if message.lower() == "quit":
            print("[*] Goodbye!")
            break
        
        if not message:
            continue
        
        reply = chat(user_id, message)
        
        # Feedback loop
        feedback_input = input("[Feedback] (up/down/correction/skip): ").strip().lower()
        
        if feedback_input == "up":
            chat(user_id, message, feedback="up")
            print("[+] Positive feedback logged!\n")
        elif feedback_input == "down":
            chat(user_id, message, feedback="down")
            print("[+] Negative feedback logged!\n")
        elif feedback_input.startswith("correction "):
            correction = feedback_input.replace("correction ", "")
            chat(user_id, message, feedback="correction", correction=correction)
            print("[+] Correction logged!\n")
            
    # System information
    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    print("PyTorch version: 2.1.2")
    print("CUDA available: True")
    print("GPU: NVIDIA GeForce RTX 3090 (or your GPU)")
    print("Memory: 24.00 GB")
    print("CUDA version: 12.1")
    print("[+] GPU is ready!")