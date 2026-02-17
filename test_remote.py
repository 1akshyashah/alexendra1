import requests
import json

BASE_URL = "https://98b5cs38-8000.inc1.devtunnels.ms"

def test_health():
    """Check server health."""
    try:
        resp = requests.get(f"{BASE_URL}/health", verify=False)
        print(f"Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

def test_chat():
    """Test chat endpoint."""
    try:
        payload = {
            "user_id": "test_user",
            "message": "Hello, how are you?",
            "feedback": None,
            "correction": None
        }
        resp = requests.post(f"{BASE_URL}/chat", json=payload, verify=False)
        print(f"Status: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("[*] Testing health endpoint...")
    test_health()
    print("\n[*] Testing chat endpoint...")
    test_chat()