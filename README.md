# Neural Network Project

Python project for building and training neural networks with **PyTorch**.

## Setup

1. **Create a virtual environment (recommended):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Project layout

| File / folder | Purpose |
|---------------|--------|
| `model.py` | Reusable `NeuralNetwork` (MLP) and `build_mnist_classifier()` |
| `train.py` | MNIST training script; downloads data, trains, saves `mnist_model.pt` |
| `requirements.txt` | PyTorch, NumPy, Matplotlib |

## Quick start

Train on MNIST (downloads dataset on first run):

```powershell
python train.py
```

## Customizing the network

- **`model.py`**  
  - `NeuralNetwork(layer_sizes, activation="relu", dropout=0.0)`  
    - `layer_sizes`: e.g. `[784, 256, 128, 10]`  
  - `build_mnist_classifier(hidden=(256, 128), dropout=0.2)`  
    - Builds 784 → hidden → 10 for MNIST.

- **`train.py`**  
  - Change `epochs`, `batch_size`, `lr`, or switch to another dataset by editing the script.

## Requirements

- Python 3.10+
- PyTorch 2.x (CPU or CUDA)
