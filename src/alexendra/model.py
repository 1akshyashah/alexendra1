"""
Simple feedforward neural network (MLP) for classification or regression.
"""
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """Multi-layer perceptron with configurable layers."""

    def __init__(self, layer_sizes, activation="relu", dropout=0.0):
        """
        Args:
            layer_sizes: List of ints, e.g. [784, 256, 128, 10] for input -> hidden -> output.
            activation: "relu", "gelu", or "tanh".
            dropout: Dropout probability (0 = no dropout).
        """
        super().__init__()
        self.layer_sizes = layer_sizes

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
        }
        self.activation = activations.get(activation, nn.ReLU())

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten if input is image-like (batch, C, H, W)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


def build_mnist_classifier(hidden=(256, 128), dropout=0.2):
    """784 -> hidden -> 10 for MNIST digit classification."""
    sizes = [784] + list(hidden) + [10]
    return NeuralNetwork(sizes, activation="relu", dropout=dropout)


class MNIST_CNN(nn.Module):
    """CNN for MNIST: more capacity, uses spatial structure."""

    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CIFAR10_CNN(nn.Module):
    """CNN for CIFAR-10: 3-channel input, more advanced task."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)


class _ResBlock(nn.Module):
    """Residual block with optional 1x1 shortcut for stride/channel change."""

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = _conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class CIFAR10_ResNet(nn.Module):
    """ResNet-style model for CIFAR-10: peak performance with residual connections."""

    def __init__(self, num_blocks=(3, 3, 3), base=64, dropout=0.3):
        super().__init__()
        self.in_c = base
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base * 4, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base * 4, 10),
        )

    def _make_layer(self, channels, blocks, stride=1):
        layers = []
        layers.append(_ResBlock(self.in_c, channels, stride))
        self.in_c = channels
        for _ in range(blocks - 1):
            layers.append(_ResBlock(channels, channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.classifier(x)
