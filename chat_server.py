import json
import time
import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()
LOG_DIR = "data"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "dialogs.jsonl")
MODEL_NAME = "distilgpt2"  # Smaller, faster model

class ChatMessage(BaseModel):
    user_id: str
    message: str
    feedback: Optional[str] = None
    correction: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    timestamp: float
    processing_time: float

def setup_gpu():
    """Configure GPU for optimal performance."""
    if not torch.cuda.is_available():
        print("[!] GPU not available! Falling back to CPU.")
        return "cpu"
    
    print(f"[+] CUDA Available: {torch.cuda.is_available()}")
    print(f"[+] GPU Count: {torch.cuda.device_count()}")
    print(f"[+] Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    return "cuda:0"

device_type = setup_gpu()
device = torch.device(device_type)

print(f"[+] Loading model '{MODEL_NAME}' on {device}...")
print("[*] This may take a moment on first run...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("[+] Model loaded successfully!\n")
except Exception as e:
    print(f"[!] Error loading model: {e}")
    print("[!] Make sure you have internet connection for downloading the model.")
    exit(1)

def append_log(entry: dict):
    """Append interaction to JSONL log."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def model_generate_reply(user_id: str, prompt: str) -> tuple:
    """Generate reply using GPU-optimized model."""
    start_time = time.time()
    
    try:
        with torch.no_grad():
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                inputs,
                max_length=150,
                num_beams=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        processing_time = time.time() - start_time
        return reply, processing_time
    
    except Exception as e:
        return f"Error generating reply: {str(e)}", time.time() - start_time

@app.post("/chat", response_model=ChatResponse)
async def chat(msg: ChatMessage) -> ChatResponse:
    """Chat endpoint: generate reply and log interaction (GPU-accelerated)."""
    response_text, processing_time = model_generate_reply(msg.user_id, msg.message)
    timestamp = time.time()
    
    entry = {
        "timestamp": timestamp,
        "datetime": datetime.fromtimestamp(timestamp).isoformat(),
        "user_id": msg.user_id,
        "user_message": msg.message,
        "bot_reply": response_text,
        "feedback": msg.feedback,
        "correction": msg.correction,
        "processing_time_ms": processing_time * 1000
    }
    append_log(entry)
    
    return ChatResponse(
        reply=response_text,
        timestamp=timestamp,
        processing_time=processing_time
    )

@app.post("/feedback")
async def submit_feedback(msg: ChatMessage):
    """Submit feedback/correction for logged interaction."""
    entry = {
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "user_id": msg.user_id,
        "feedback_type": msg.feedback,
        "correction": msg.correction,
        "original_message": msg.message
    }
    append_log(entry)
    return {"status": "feedback logged"}

@app.get("/health")
async def health():
    """Health check with GPU status."""
    try:
        if "cuda" in str(device):
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_memory_max = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            return {
                "status": "ok",
                "model": MODEL_NAME,
                "device": "cuda:0",
                "gpu_memory_used_gb": f"{gpu_memory:.2f}",
                "gpu_memory_total_gb": f"{gpu_memory_max:.2f}",
                "cuda_available": True
            }
        else:
            return {
                "status": "ok",
                "model": MODEL_NAME,
                "device": "cpu",
                "cuda_available": False
            }
    except:
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "device": "cpu",
            "cuda_available": False
        }

@app.get("/gpu-stats")
async def gpu_stats():
    """Get detailed GPU statistics."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    else:
        return {"error": "CUDA not available"}

if __name__ == "__main__":
    import uvicorn
    print("\n[+] Starting GPU-accelerated chat server...")
    print("[+] Server running on http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)