"""
Visualize neural network input and output (predictions + class probabilities).
Supports MNIST and CIFAR-10.

Run:
    python visualize.py
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model import build_mnist_classifier, MNIST_CNN, CIFAR10_CNN, CIFAR10_ResNet

CIFAR10_CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


# ---------------------------------------------------
# Helper: Clean compiled prefix
# ---------------------------------------------------
def clean_state_dict(state_dict):
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace("_orig_mod.", "")] = v
        return new_state
    return state_dict


# ---------------------------------------------------
# Auto-detect base width from checkpoint
# ---------------------------------------------------
def detect_base_from_checkpoint(state_dict):
    """
    Detect base channel size from first conv layer.
    """
    for k, v in state_dict.items():
        if "conv1.0.weight" in k:
            # Shape: [base, 3, 3, 3]
            return v.shape[0]
    return 64  # fallback default


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- CIFAR10 RESNET ----------------
    if os.path.exists("cifar10_resnet_model.pt"):
        print("Loading CIFAR10 ResNet...")

        state_dict = torch.load("cifar10_resnet_model.pt", map_location=device)
        state_dict = clean_state_dict(state_dict)

        # Auto detect base
        base = detect_base_from_checkpoint(state_dict)
        print(f"Detected base width: {base}")

        model = CIFAR10_ResNet(num_blocks=(3, 3, 3), base=base, dropout=0.3).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        norm = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )

        test_data = datasets.CIFAR10(
            "./data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                norm
            ])
        )

        class_names = list(CIFAR10_CLASSES)

        def denorm(t):
            return (
                t * torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
                + torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            )

    # ---------------- CIFAR10 CNN ----------------
    elif os.path.exists("cifar10_cnn_model.pt"):
        print("Loading CIFAR10 CNN...")

        state_dict = torch.load("cifar10_cnn_model.pt", map_location=device)
        state_dict = clean_state_dict(state_dict)

        model = CIFAR10_CNN(dropout=0.3).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        norm = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )

        test_data = datasets.CIFAR10(
            "./data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                norm
            ])
        )

        class_names = list(CIFAR10_CLASSES)

        def denorm(t):
            return (
                t * torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
                + torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            )

    # ---------------- MNIST CNN ----------------
    elif os.path.exists("mnist_cnn_model.pt"):
        print("Loading MNIST CNN...")

        state_dict = torch.load("mnist_cnn_model.pt", map_location=device)
        state_dict = clean_state_dict(state_dict)

        model = MNIST_CNN(dropout=0.25).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        test_data = datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        class_names = [str(i) for i in range(10)]
        denorm = None

    # ---------------- MNIST MLP ----------------
    else:
        print("Loading MNIST MLP...")

        state_dict = torch.load("mnist_model.pt", map_location=device)
        state_dict = clean_state_dict(state_dict)

        model = build_mnist_classifier(hidden=(256, 128), dropout=0.2).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        test_data = datasets.MNIST(
            "./data",
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )

        class_names = [str(i) for i in range(10)]
        denorm = None

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=True)
    images, labels = next(iter(loader))

    if denorm is not None:
        images = denorm(images.clone())

    with torch.no_grad():
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu()

    n = min(12, images.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(6, 2 * n),
                             gridspec_kw={"width_ratios": [1, 2]})

    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        ax_img, ax_out = axes[i, 0], axes[i, 1]

        if images.shape[1] == 3:
            ax_img.imshow(images[i].permute(1, 2, 0).numpy().clip(0, 1))
        else:
            ax_img.imshow(images[i].squeeze().numpy(), cmap="gray")

        ax_img.set_xticks([])
        ax_img.set_yticks([])

        true_label = labels[i].item()
        pred_label = preds[i].item()
        prob = probs[i].cpu().numpy()

        ax_img.set_title(f"True: {class_names[true_label]}", fontsize=10)
        color = "green" if pred_label == true_label else "red"
        ax_img.set_ylabel(f"Pred: {class_names[pred_label]}",
                          fontsize=10, color=color)

        ax_out.bar(range(len(class_names)),
                   prob,
                   color=["green" if j == pred_label else "steelblue"
                          for j in range(len(class_names))])

        ax_out.set_xticks(range(len(class_names)))
        ax_out.set_xticklabels(class_names, rotation=45, ha="right")
        ax_out.set_ylabel("Probability")
        ax_out.set_ylim(0, 1)

    plt.suptitle("Input → Output (class probabilities)", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()