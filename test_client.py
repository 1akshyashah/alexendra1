import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def chat(user_id: str, message: str):
    """Send a chat message and get a reply."""
    payload = {
        "user_id": user_id,
        "message": message,
        "feedback": None,
        "correction": None
    }
    resp = requests.post(f"{BASE_URL}/chat", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(f"\n[Bot]: {data['reply']}")
        print(f"Processing time: {data['processing_time']:.3f}s")
        return data['reply']
    else:
        print(f"Error: {resp.status_code}")

def submit_feedback(user_id: str, message: str, feedback: str, correction: str = None):
    """Submit feedback or correction."""
    payload = {
        "user_id": user_id,
        "message": message,
        "feedback": feedback,
        "correction": correction
    }
    resp = requests.post(f"{BASE_URL}/feedback", json=payload)
    print(f"Feedback submitted: {resp.json()}")

if __name__ == "__main__":
    print("[*] Testing chat server...\n")
    
    user_id = "user_01"
    print(f"[You]: Hello, how are you?")
    chat(user_id, "Hello, how are you?")
    time.sleep(1)
    
    print(f"\n[You]: What is AI?")
    chat(user_id, "What is AI?")
    time.sleep(1)
    
    print("\n[*] Submitting positive feedback...")
    submit_feedback(user_id, "What is AI?", "up")
    
    print("\n[*] Dialog log:")
    if os.path.exists("data/dialogs.jsonl"):
        with open("data/dialogs.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                print(f"\n--- Entry {i+1} ---")
                print(json.dumps(rec, indent=2))