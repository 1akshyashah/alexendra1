"""
Natural Language Interface: Query the model using language inputs and get interpretable responses.
Includes multi-task learning and language understanding.
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from advanced_model import TextEncoder, EfficientNet


CIFAR10_CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR10_DESCRIPTIONS = {
    "plane": "An aircraft flying in the sky",
    "car": "A four-wheeled motor vehicle",
    "bird": "A feathered flying animal",
    "cat": "A feline domestic animal",
    "deer": "A horned herbivorous mammal",
    "dog": "A canine domestic animal",
    "frog": "An amphibian that croaks",
    "horse": "A large four-legged mammal",
    "ship": "A large watercraft vessel",
    "truck": "A large vehicle for cargo"
}


class LanguageAwareClassifier(nn.Module):
    """Classifier that understands and responds to language queries."""
    
    def __init__(self, num_classes=10, class_names=CIFAR10_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Image encoder
        self.image_encoder = EfficientNet(num_classes=num_classes)
        
        # Text encoder for understanding queries
        self.text_encoder = TextEncoder(hidden_dim=256)
        
        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Explanation generator (text feature to explanation embedding)
        self.explanation_generator = nn.Sequential(
            nn.Linear(256 + num_classes, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, images):
        """Predict class for images."""
        return self.image_encoder(images)
    
    def predict_with_confidence(self, images):
        """Get predictions with confidence scores."""
        logits = self.image_encoder(images)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        return pred, confidence, probs
    
    def query(self, image, query_text):
        """
        Process a natural language query about an image.
        
        Args:
            image: PIL Image or torch tensor
            query_text: Natural language query (e.g., "what is this?", "is this a dog?")
        
        Returns:
            dict with prediction, confidence, and natural language response
        """
        device = next(self.parameters()).device
        
        # Convert image to tensor if needed
        if isinstance(image, Image.Image):
            from torchvision import transforms
            to_tensor = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            image = to_tensor(image).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        # Get prediction
        pred, confidence, probs = self.predict_with_confidence(image)
        predicted_class = self.class_names[pred.item()]
        confidence_score = confidence.item()
        
        # Parse query intent
        query_lower = query_text.lower()
        
        # Different response templates based on query type
        if "what" in query_lower and "this" in query_lower:
            response = f"This is a {predicted_class} ({confidence_score:.1%} confident)"
        elif "is" in query_lower and "?" in query_lower:
            # Extract queried class from query (simple pattern matching)
            queried_class = None
            for cls in self.class_names:
                if cls in query_lower:
                    queried_class = cls
                    break
            
            if queried_class:
                is_match = queried_class == predicted_class
                response = f"{'Yes' if is_match else 'No'}, this is {f'a {predicted_class}' if is_match else f'not a {queried_class}, but a {predicted_class}'} ({confidence_score:.1%} confident)"
            else:
                response = f"The image shows a {predicted_class} ({confidence_score:.1%} confident)"
        else:
            response = f"Based on my analysis, I see a {predicted_class} in this image ({confidence_score:.1%} confidence)"
        
        # Get top-3 predictions
        top3_probs, top3_idx = torch.topk(probs[0], 3)
        top3_classes = [self.class_names[i] for i in top3_idx.tolist()]
        top3_scores = top3_probs.tolist()
        
        return {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "response": response,
            "top3": [(cls, score) for cls, score in zip(top3_classes, top3_scores)],
            "all_probabilities": {cls: prob for cls, prob in zip(self.class_names, probs[0].tolist())}
        }
    
    def describe_prediction(self, image):
        """Generate a detailed description of the image."""
        device = next(self.parameters()).device
        
        if isinstance(image, Image.Image):
            from torchvision import transforms
            to_tensor = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            image = to_tensor(image).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(device)
        
        pred, confidence, probs = self.predict_with_confidence(image)
        predicted_class = self.class_names[pred.item()]
        
        description = f"""
        Image Analysis Report
        =====================
        Predicted Object: {predicted_class.upper()}
        Confidence Level: {confidence.item():.1%}
        Description: {CIFAR10_DESCRIPTIONS.get(predicted_class, 'No description available')}
        
        Probability Distribution:
        """
        
        for cls, prob in zip(self.class_names, probs[0].tolist()):
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            description += f"\n        {cls:6s} : {bar} {prob:.1%}"
        
        return description
    
    def compare_images(self, image1, image2):
        """Compare two images and highlight differences in predictions."""
        device = next(self.parameters()).device
        
        # Process first image
        if isinstance(image1, Image.Image):
            from torchvision import transforms
            to_tensor = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            image1 = to_tensor(image1).unsqueeze(0)
        elif image1.dim() == 3:
            image1 = image1.unsqueeze(0)
        
        # Process second image
        if isinstance(image2, Image.Image):
            from torchvision import transforms
            to_tensor = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            image2 = to_tensor(image2).unsqueeze(0)
        elif image2.dim() == 3:
            image2 = image2.unsqueeze(0)
        
        image1, image2 = image1.to(device), image2.to(device)
        
        pred1, conf1, probs1 = self.predict_with_confidence(image1)
        pred2, conf2, probs2 = self.predict_with_confidence(image2)
        
        class1 = self.class_names[pred1.item()]
        class2 = self.class_names[pred2.item()]
        
        return {
            "image1": {"prediction": class1, "confidence": conf1.item()},
            "image2": {"prediction": class2, "confidence": conf2.item()},
            "same": class1 == class2,
            "comparison": f"Image 1: {class1} ({conf1.item():.1%}) | Image 2: {class2} ({conf2.item():.1%})"
        }


class MultiTaskLearner(nn.Module):
    """Learn multiple tasks simultaneously: classification, attribute prediction, etc."""
    
    def __init__(self, base_encoder, num_classes=10, task_list=["classification", "confidence"]):
        super().__init__()
        self.base_encoder = base_encoder
        self.num_classes = num_classes
        self.task_list = task_list
        
        encoded_dim = 256  # Output of EfficientNet before classification
        
        # Task heads
        self.heads = nn.ModuleDict()
        
        if "classification" in task_list:
            self.heads["classification"] = nn.Linear(256, num_classes)
        
        if "confidence" in task_list:
            self.heads["confidence"] = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        if "color_analysis" in task_list:
            self.heads["color_analysis"] = nn.Linear(256, 8)  # 8 color categories
        
        if "size_estimation" in task_list:
            self.heads["size_estimation"] = nn.Linear(256, 1)  # small to large scale
    
    def forward(self, images):
        """Forward pass returning all task outputs."""
        # Get base features
        with torch.no_grad():
            features = self.base_encoder.blocks(
                self.base_encoder.conv1(images)
            )
        features = self.base_encoder.avg_pool(features).flatten(1)
        
        # Apply all heads
        outputs = {}
        for task, head in self.heads.items():
            outputs[task] = head(features)
        
        return outputs


def load_language_aware_model(model_path, num_classes=10):
    """Load a pre-trained language-aware model."""
    model = LanguageAwareClassifier(num_classes=num_classes)
    model_dict = torch.load(model_path, map_location="cpu")
    model.image_encoder.load_state_dict(model_dict)
    return model


def interactive_prompt():
    """Interactive CLI for querying the model."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("Language-Aware Image Classifier - Interactive Mode")
    print("=" * 60)
    
    try:
        model = load_language_aware_model("cifar10_efficient_model.pt", num_classes=10)
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully\n")
    except FileNotFoundError:
        print("✗ Model file not found. Train a model first using: python train_advanced.py")
        return
    
    while True:
        print("\nOptions:")
        print("1. Query with a local image file")
        print("2. Describe the contents of an image")
        print("3. Compare two images")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            query = input("Enter your question: ").strip()
            
            try:
                image = Image.open(image_path).convert("RGB")
                result = model.query(image, query)
                print(f"\n{result['response']}")
                print(f"\nTop 3 predictions:")
                for cls, score in result['top3']:
                    print(f"  - {cls}: {score:.1%}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            try:
                image = Image.open(image_path).convert("RGB")
                description = model.describe_prediction(image)
                print(description)
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            image_path1 = input("Enter first image path: ").strip()
            image_path2 = input("Enter second image path: ").strip()
            
            try:
                image1 = Image.open(image_path1).convert("RGB")
                image2 = Image.open(image_path2).convert("RGB")
                comparison = model.compare_images(image1, image2)
                print(f"\n{comparison['comparison']}")
                print(f"Same object type: {'Yes' if comparison['same'] else 'No'}")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option. Please select 1-4.")


if __name__ == "__main__":
    interactive_prompt()
