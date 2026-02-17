"""
Peak-performance training: CIFAR-10 ResNet, LR scheduler, gradient clipping.
Graph visualizations: loss curve, accuracy curve, confusion matrix.
"""
import os
import warnings

# Suppress NumPy 2.4 + torchvision CIFAR pickle deprecation (cifar.py line 83)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision.datasets.cifar")
warnings.filterwarnings("ignore", message=".*align.*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from model import build_mnist_classifier, MNIST_CNN, CIFAR10_CNN, CIFAR10_ResNet

# ---------- Config: peak performance & harder task ----------
DATASET = "cifar10"           # "cifar10" (harder) or "mnist"
MODEL_TYPE = "resnet"         # "resnet" (peak), "cnn", or "mlp" (mnist only)
BATCH_SIZE = 64               # 64 for faster CPU training
EPOCHS = 50                   # convergence achieved by epoch 50 on CIFAR-10
NUM_WORKERS = 4
PIN_MEMORY = True
LR = 0.1                      # initial LR for ResNet + cosine schedule
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 1.0               # gradient clipping
SHOW_SAMPLE_VIZ = True        # show sample predictions at end (not every epoch)
# -----------------------------------------------------------


CIFAR10_CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=PIN_MEMORY), target.to(device, non_blocking=PIN_MEMORY)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device, non_blocking=PIN_MEMORY), target.to(device, non_blocking=PIN_MEMORY)
        output = model(data)
        total_loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def get_confusion_matrix(model, loader, device, num_classes=10):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for data, target in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1).cpu()
        target_cpu = target.cpu()
        for t, p in zip(target_cpu.tolist(), pred.tolist()):
            cm[int(t), int(p)] += 1
    return cm.numpy()


def plot_training_curves(history, save_dir="."):
    """Plot loss and accuracy curves; save to file."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="C0")
    ax1.plot(epochs, history["test_loss"], label="Test loss", color="C1")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train acc", color="C0")
    ax2.plot(epochs, history["test_acc"], label="Test acc", color="C1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {path}")


def plot_confusion_matrix(cm, class_names, save_dir="."):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {path}")


def visualize_sample_batch(model, images, labels, device, dataset_name, class_names=None):
    """Show one grid of inputs and predictions."""
    model.eval()
    with torch.no_grad():
        preds = model(images.to(device)).argmax(dim=1).cpu()
    n = min(8, images.size(0))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.8))
    for i in range(n):
        img = images[i]
        if img.shape[0] == 3:
            axes[0, i].imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
        else:
            axes[0, i].imshow(img.squeeze().numpy(), cmap="gray")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        true_l = labels[i].item()
        pred_l = preds[i].item()
        axes[0, i].set_title(f"True: {class_names[true_l] if class_names else true_l}", fontsize=9)
        c = "green" if pred_l == true_l else "red"
        axes[1, i].text(0.5, 0.5, f"Pred: {class_names[pred_l] if class_names else pred_l}", ha="center", va="center", fontsize=11, color=c)
        axes[1, i].axis("off")
    plt.suptitle("Sample predictions: Input (top) → Output (bottom)")
    plt.tight_layout()
    plt.show()


def main():
    device = get_device()
    use_cuda = device.type == "cuda"
    pin_memory = PIN_MEMORY and use_cuda
    num_workers = NUM_WORKERS
    if os.name == "nt" and num_workers > 0:
        num_workers = min(num_workers, 2)
    print(f"Device: {device}  |  batch={BATCH_SIZE}  workers={num_workers}  epochs={EPOCHS}  dataset={DATASET}  model={MODEL_TYPE}")

    if DATASET == "cifar10":
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), norm])
        train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10("./data", train=False, transform=transform_test)
        class_names = list(CIFAR10_CLASSES)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_data = datasets.MNIST("./data", train=False, transform=transform)
        class_names = [str(i) for i in range(10)]

    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
    train_loader = DataLoader(train_data, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kw)

    if DATASET == "cifar10":
        if MODEL_TYPE == "resnet":
            model = CIFAR10_ResNet(num_blocks=(3, 3, 3), base=32, dropout=0.3).to(device)
        else:
            model = CIFAR10_CNN(dropout=0.3).to(device)
    elif MODEL_TYPE == "cnn":
        model = MNIST_CNN(dropout=0.25).to(device)
    else:
        model = build_mnist_classifier(hidden=(256, 128), dropout=0.2).to(device)

    try:
        model = torch.compile(model, backend="aot_eager")
        print("[+] Model compiled with torch.compile()")
    except Exception as e:
        print(f"[!] torch.compile() not available: {e}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    ) if MODEL_TYPE == "resnet" else torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) if MODEL_TYPE == "resnet" else None
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=GRAD_CLIP)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{EPOCHS}  train loss: {train_loss:.4f}  train acc: {train_acc:.4f}  test acc: {test_acc:.4f}  lr: {lr:.2e}")
    plt.ioff()

    ckpt = f"{DATASET}_{MODEL_TYPE}_model.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved model to {ckpt}  |  best test acc: {best_acc:.4f}")

    # ---------- Graph visualizations ----------
    plot_training_curves(history)
    cm = get_confusion_matrix(model, test_loader, device, num_classes=10)
    plot_confusion_matrix(cm, class_names)
    if SHOW_SAMPLE_VIZ:
        viz_loader = DataLoader(test_data, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(99))
        viz_imgs, viz_labels = next(iter(viz_loader))
        if DATASET == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
            viz_imgs = viz_imgs * std + mean
        visualize_sample_batch(model, viz_imgs, viz_labels, device, DATASET, class_names)


if __name__ == "__main__":
    main()
