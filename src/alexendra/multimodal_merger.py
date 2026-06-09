import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from torchvision import transforms

class MultiModalLearner:
    """Merges image, number, and internet data learning."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.text_model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(self.device)
        
        # Placeholder for image model
        self.image_model = None
        
        # Placeholder for number predictor
        self.number_model = None
        
        self.fusion_weights = {
            "text": 0.5,
            "image": 0.25,
            "number": 0.25
        }
        
        print("[+] MultiModal Learner initialized")
    
    def load_internet_data(self, crawl_file: str):
        """Load internet crawled data."""
        internet_examples = []
        
        if not os.path.exists(crawl_file):
            print(f"[!] Crawl file not found: {crawl_file}")
            return internet_examples
        
        with open(crawl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec['type'] == 'code':
                        internet_examples.append({
                            "source": "internet",
                            "text": f"Learn from {rec['source']}: coding best practices",
                            "type": "code"
                        })
                    elif rec['type'] == 'article':
                        internet_examples.append({
                            "source": "internet",
                            "text": f"Learn from {rec['source']}: programming knowledge",
                            "type": "article"
                        })
                except json.JSONDecodeError:
                    continue
        
        print(f"[+] Loaded {len(internet_examples)} internet examples")
        return internet_examples
    
    def load_chat_data(self, chat_file: str):
        """Load chat/feedback data."""
        chat_examples = []
        
        if not os.path.exists(chat_file):
            return chat_examples
        
        with open(chat_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("feedback") == "up" or rec.get("correction"):
                        chat_examples.append({
                            "source": "chat",
                            "user": rec.get("user_message", ""),
                            "bot": rec.get("bot_reply", ""),
                            "correction": rec.get("correction"),
                            "type": "dialogue"
                        })
                except json.JSONDecodeError:
                    continue
        
        print(f"[+] Loaded {len(chat_examples)} chat examples")
        return chat_examples
    
    def load_image_data(self, image_dir: str = "data/images"):
        """Load image data and extract features."""
        image_examples = []
        
        if not os.path.exists(image_dir):
            print(f"[!] Image directory not found: {image_dir}")
            return image_examples
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        for img_file in os.listdir(image_dir):
            if img_file.endswith(('.jpg', '.png')):
                try:
                    img = Image.open(os.path.join(image_dir, img_file))
                    img_tensor = transform(img).to(self.device)
                    
                    image_examples.append({
                        "source": "image",
                        "filename": img_file,
                        "tensor": img_tensor,
                        "type": "visual"
                    })
                except Exception as e:
                    print(f"[!] Error loading {img_file}: {e}")
        
        print(f"[+] Loaded {len(image_examples)} image examples")
        return image_examples
    
    def load_number_data(self, number_file: str = "data/numbers.json"):
        """Load number prediction data."""
        number_examples = []
        
        if not os.path.exists(number_file):
            print(f"[!] Number file not found: {number_file}")
            return number_examples
        
        try:
            with open(number_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                number_examples = data.get("examples", [])
        except:
            pass
        
        print(f"[+] Loaded {len(number_examples)} number examples")
        return number_examples
    
    def fuse_modalities(self, text_data, image_data, number_data):
        """Fuse all 3 modalities into unified representation."""
        fused = {
            "text_weight": self.fusion_weights["text"],
            "image_weight": self.fusion_weights["image"],
            "number_weight": self.fusion_weights["number"],
            "combined_examples": []
        }
        
        for ex in text_data[:len(text_data) // 2]:
            fused["combined_examples"].append({
                "source": "text",
                "content": ex,
                "weight": self.fusion_weights["text"]
            })
        
        for ex in image_data[:len(image_data) // 2]:
            fused["combined_examples"].append({
                "source": "image",
                "content": ex.get('filename', 'unknown'),
                "weight": self.fusion_weights["image"]
            })
        
        for ex in number_data[:len(number_data) // 2]:
            fused["combined_examples"].append({
                "source": "number",
                "content": ex,
                "weight": self.fusion_weights["number"]
            })
        
        print(f"[+] Fused {len(fused['combined_examples'])} examples from all modalities")
        return fused
    
    def generate_merged_training_data(self):
        """Generate training data from all sources."""
        print("\n[*] Loading data from all modalities...")
        
        internet_data = self.load_internet_data("data/internet_data/crawl_latest.jsonl") if os.path.exists("data/internet_data") else []
        chat_data = self.load_chat_data("data/dialogs.jsonl")
        image_data = self.load_image_data()
        number_data = self.load_number_data()
        
        text_data = internet_data + chat_data
        
        fused = self.fuse_modalities(text_data, image_data, number_data)
        
        merged_file = "data/merged_training.json"
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(fused, f, indent=2, default=str)
        
        print(f"\n[+] Merged training data saved to {merged_file}")
        return fused