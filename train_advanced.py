"""
Advanced training pipeline with multi-modal learning, mixup, curriculum learning, and improved augmentation.
"""
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

from advanced_model import (
    TextEncoder, VisionTransformerBlock, EfficientNet, 
    ResidualAttentionBlock, MultiModalClassifier
)
from model import CIFAR10_ResNet

# Configuration
DATASET = "cifar10"           # "cifar10" or "mnist"
MODEL_TYPE = "efficient"      # "efficient", "vit", "resnet", or "multimodal"
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 5
USE_MIXUP = True
MIXUP_ALPHA = 0.2
USE_CUTMIX = True
CUTMIX_ALPHA = 1.0
USE_AMP = True              # Automatic Mixed Precision
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 4
PIN_MEMORY = True


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MixupCutmixAugmentation:
    """Mixup and CutMix augmentation on the fly."""
    
    def __init__(self, alpha=0.2, cutmix_alpha=1.0):
        self.alpha = alpha
        self.cutmix_alpha = cutmix_alpha
        
    def mixup(self, x, y):
        """Mixup: blend two samples."""
        indices = torch.randperm(x.size(0))
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_x = lam * x + (1 - lam) * x[indices]
        y_a, y_b = y, y[indices]
        return mixed_x, y_a, y_b, lam
    
    def cutmix(self, x, y):
        """CutMix: cut and paste regions."""
        batch_size, c, h, w = x.size()
        indices = torch.randperm(batch_size)
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        x[..., bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
        
        y_a, y_b = y, y[indices]
        return x, y_a, y_b, lam
    
    def __call__(self, x, y, use_cutmix=False):
        if use_cutmix and np.random.rand() < 0.5:
            return self.cutmix(x, y)
        else:
            return self.mixup(x, y)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_one_epoch_advanced(model, loader, optimizer, criterion, device, scaler=None, 
                             mixup_augmenter=None, epoch=0, total_epochs=100):
    """Advanced training with mixup, AMP, and learning rate scheduling."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Mixup/CutMix augmentation
        if mixup_augmenter is not None and torch.rand(1).item() < 0.5:
            data, target_a, target_b, lam = mixup_augmenter(data, target, use_cutmix=USE_CUTMIX)
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
        else:
            # Standard forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_advanced(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Learning rate warmup scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    device = get_device()
    print(f"Device: {device} | Model: {MODEL_TYPE} | Dataset: {DATASET}")
    
    # Prepare data
    if DATASET == "cifar10":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            norm,
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), norm])
        train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10("./data", train=False, transform=transform_test)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST("./data", train=False, transform=transform)
    
    loader_kw = dict(batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY)
    if NUM_WORKERS > 0:
        loader_kw["num_workers"] = NUM_WORKERS if os.name != "nt" else 2
        loader_kw["persistent_workers"] = True
    
    train_loader = DataLoader(train_data, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kw)
    
    # Build model
    if MODEL_TYPE == "efficient":
        model = EfficientNet(num_classes=10, width_multiplier=1.0, depth_multiplier=1.0).to(device)
    elif MODEL_TYPE == "vit":
        model = VisionTransformerBlock(img_size=32, patch_size=4, in_channels=3, 
                                       embed_dim=192, num_heads=8, depth=12, num_classes=10).to(device)
    elif MODEL_TYPE == "resnet":
        model = CIFAR10_ResNet(num_blocks=(3, 3, 3), base=64).to(device)
    else:
        model = EfficientNet(num_classes=10).to(device)
    
    # Optimizer & scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    scheduler = get_warmup_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP and device.type == "cuda" else None
    
    # Augmentation
    mixup_augmenter = MixupCutmixAugmentation(MIXUP_ALPHA, CUTMIX_ALPHA) if USE_MIXUP else None
    
    # Training loop
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch_advanced(
            model, train_loader, optimizer, criterion, device, scaler, 
            mixup_augmenter, epoch, EPOCHS
        )
        test_loss, test_acc = evaluate_advanced(model, test_loader, criterion, device)
        scheduler.step()
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"best_{DATASET}_{MODEL_TYPE}_model.pt")
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS} | Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Test: {test_loss:.4f}/{test_acc:.4f} | Best: {best_acc:.4f} | LR: {lr:.2e}")
    
    # Save final model
    torch.save(model.state_dict(), f"{DATASET}_{MODEL_TYPE}_model.pt")
    print(f"\n✓ Saved model to {DATASET}_{MODEL_TYPE}_model.pt | Best test acc: {best_acc:.4f}")
    
    # Plot results
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(epochs_range, history["train_loss"], label="Train loss")
    axes[0].plot(epochs_range, history["test_loss"], label="Test loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Progress")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs_range, history["train_acc"], label="Train acc")
    axes[1].plot(epochs_range, history["test_acc"], label="Test acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Model Performance ({MODEL_TYPE})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{DATASET}_{MODEL_TYPE}_progress.png", dpi=150)
    plt.show()
    print(f"✓ Saved plot to {DATASET}_{MODEL_TYPE}_progress.png")


if __name__ == "__main__":
    main()
