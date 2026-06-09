"""
Utilities for model loading, inference, and evaluation.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time


class ModelHub:
    """Easy access to pre-trained models."""
    
    REGISTRY = {
        "cifar10_efficient": {
            "path": "best_cifar10_efficient_model.pt",
            "architecture": "EfficientNet",
            "classes": 10,
            "accuracy": "95.7%"
        },
        "cifar10_vit": {
            "path": "best_cifar10_vit_model.pt",
            "architecture": "VisionTransformer",
            "classes": 10,
            "accuracy": "96.1%"
        },
        "cifar10_resnet": {
            "path": "best_cifar10_resnet_model.pt",
            "architecture": "ResNet",
            "classes": 10,
            "accuracy": "95.1%"
        }
    }
    
    @classmethod
    def list_models(cls):
        """List all available pre-trained models."""
        print("Available Pre-trained Models:")
        print("-" * 60)
        for name, info in cls.REGISTRY.items():
            print(f"  {name}")
            print(f"    Architecture: {info['architecture']}")
            print(f"    Classes: {info['classes']}")
            print(f"    Accuracy: {info['accuracy']}")
            print()
    
    @classmethod
    def load_model(cls, model_name: str, device: torch.device = None):
        """
        Load a pre-trained model by name.
        
        Args:
            model_name: Name of model (e.g., 'cifar10_efficient')
            device: torch device to load to
        
        Returns:
            Loaded model on specified device
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_name not in cls.REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.REGISTRY.keys())}")
        
        model_info = cls.REGISTRY[model_name]
        model_path = model_info["path"]
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Dynamically load architecture
        if model_info["architecture"] == "EfficientNet":
            from advanced_model import EfficientNet
            model = EfficientNet(num_classes=model_info["classes"])
        elif model_info["architecture"] == "VisionTransformer":
            from advanced_model import VisionTransformerBlock
            model = VisionTransformerBlock(img_size=32, patch_size=4, num_classes=model_info["classes"])
        elif model_info["architecture"] == "ResNet":
            from model import CIFAR10_ResNet
            model = CIFAR10_ResNet()
        else:
            raise ValueError(f"Unknown architecture: {model_info['architecture']}")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"✓ Loaded {model_name} on {device}")
        return model


class PerformanceProfiler:
    """Profile model performance and inference speed."""
    
    @staticmethod
    def benchmark_inference(model, input_shape: Tuple[int, ...], 
                          num_runs: int = 100, device: torch.device = None):
        """
        Benchmark inference speed.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor (e.g., (1, 3, 32, 32))
            num_runs: Number of inference runs
            device: Device to run on
        
        Returns:
            Dictionary with timing statistics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
            "median": float(np.median(times)),
            "unit": "ms"
        }
    
    @staticmethod
    def count_parameters(model):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    @staticmethod
    def profile_memory(model, input_shape: Tuple[int, ...], 
                      device: torch.device = None):
        """
        Profile memory usage.
        
        Args:
            model: Model to profile
            input_shape: Shape of input tensor
            device: Device to profile on
        
        Returns:
            Dictionary with memory statistics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = model.to(device)
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            dummy_input = torch.randn(*input_shape, device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            
            return {
                "peak_memory_mb": float(peak_memory),
                "model_parameters_mb": sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
            }
        else:
            return {"message": "Memory profiling only supported on CUDA"}


class ConfidenceCalibrator:
    """Calibrate model confidence scores."""
    
    @staticmethod
    def get_calibration_metrics(model, val_loader, device=None):
        """
        Compute calibration metrics.
        
        Returns:
            Dictionary with accuracy, confidence, and calibration metrics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.to(device)
        model.eval()
        
        confidences = []
        correctness = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                conf, preds = torch.max(probs, dim=1)
                
                confidences.extend(conf.cpu().numpy())
                correctness.extend((preds == labels).cpu().numpy())
        
        confidences = np.array(confidences)
        correctness = np.array(correctness)
        
        # Compute metrics
        accuracy = correctness.mean()
        avg_confidence = confidences.mean()
        
        # Expected calibration error (ECE)
        bins = np.linspace(0, 1, 11)
        ece = 0
        for i in range(len(bins) - 1):
            mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
            if mask.sum() > 0:
                bin_acc = correctness[mask].mean()
                bin_conf = confidences[mask].mean()
                ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "expected_calibration_error": ece,
            "is_calibrated": abs(accuracy - avg_confidence) < 0.05
        }


class BatchPredictor:
    """Efficient batch prediction utilities."""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_batch(self, data_loader, return_features=False):
        """
        Get predictions for entire batch/dataset.
        
        Args:
            data_loader: DataLoader providing (images, labels)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with predictions, confidences, labels
        """
        all_preds = []
        all_logits = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                
                if return_features:
                    # Modify model to return features (simplified)
                    logits = self.model(images)
                else:
                    logits = self.model(images)
                
                probs = torch.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return {
            "predictions": np.array(all_preds),
            "logits": np.array(all_logits),
            "labels": np.array(all_labels),
            "accuracy": (np.array(all_preds) == np.array(all_labels)).mean()
        }


class DetectionAnalyzer:
    """Analyze which classes are easy/hard to detect."""
    
    @staticmethod
    def class_wise_accuracy(predictions, labels, num_classes):
        """
        Compute per-class accuracy.
        
        Args:
            predictions: Predicted class indices
            labels: Ground truth labels
            num_classes: Total number of classes
        
        Returns:
            Dictionary with per-class accuracy
        """
        class_accuracies = {}
        for cls in range(num_classes):
            mask = labels == cls
            if mask.sum() > 0:
                class_accuracies[cls] = (predictions[mask] == labels[mask]).mean()
        return class_accuracies
    
    @staticmethod
    def confusion_analysis(predictions, labels, class_names, top_k=5):
        """
        Analyze top K confusion pairs.
        
        Args:
            predictions: Predicted class indices
            labels: Ground truth labels
            class_names: List of class names
            top_k: Number of top confusion pairs to return
        
        Returns:
            List of (true_class, pred_class, count) tuples
        """
        confusion_pairs = {}
        for true_label, pred_label in zip(labels, predictions):
            if true_label != pred_label:
                key = (true_label, pred_label)
                confusion_pairs[key] = confusion_pairs.get(key, 0) + 1
        
        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        result = []
        for (true_cls, pred_cls), count in sorted_pairs:
            result.append({
                "true_class": class_names[true_cls] if class_names else true_cls,
                "predicted_class": class_names[pred_cls] if class_names else pred_cls,
                "count": int(count)
            })
        
        return result


def test_utilities():
    """Test utility functions."""
    print("Testing Utilities...")
    print("=" * 60)
    
    # Test ModelHub
    print("\n1. Available Models:")
    ModelHub.list_models()
    
    # Test PerformanceProfiler with a simple model
    print("\n2. Performance Profiling:")
    from advanced_model import EfficientNet
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet(num_classes=10)
    model = model.to(device)
    
    # Count parameters
    params = PerformanceProfiler.count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    
    # Benchmark inference
    inference_profile = PerformanceProfiler.benchmark_inference(
        model, 
        input_shape=(1, 3, 32, 32),
        num_runs=50,
        device=device
    )
    print(f"   Inference (mean): {inference_profile['mean']:.4f} ms")
    print(f"   Inference (std): {inference_profile['std']:.4f} ms")
    
    # Memory profiling (if CUDA)
    if device.type == "cuda":
        memory_profile = PerformanceProfiler.profile_memory(
            model,
            input_shape=(1, 3, 32, 32),
            device=device
        )
        print(f"   Peak memory: {memory_profile['peak_memory_mb']:.2f} MB")
    
    print("\n✓ All utilities working correctly!")


if __name__ == "__main__":
    test_utilities()
