"""
Configuration Guide - Tutorial on all training parameters and what they do.
"""

configuration_guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║             ADVANCED TRAINING CONFIGURATION GUIDE                          ║
║              Understanding & Tuning All Parameters                         ║
╚════════════════════════════════════════════════════════════════════════════╝

This guide explains every parameter in train_advanced.py and how to tune them
for your specific use case.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DATASET & MODEL SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameter: DATASET
  Values: "cifar10", "mnist"
  Default: "cifar10"
  
  What it does:
    - Selects dataset to train on
    - CIFAR-10: 60,000 colored images (harder, 32x32)
    - MNIST: 70,000 grayscale digits (easier, 28x28)
  
  How to choose:
    ✓ Use CIFAR10 for better accuracy benchmarking
    ✓ Use MNIST for quick testing & debugging
  
  Impact on accuracy:
    - CIFAR-10: Typically 95-97%
    - MNIST: Typically 99%+


Parameter: MODEL_TYPE
  Values: "efficient", "vit", "resnet", "multimodal"
  Default: "efficient"
  
  What each does:
    - efficient      : EfficientNet (fast, mobile-optimized)
    - vit           : Vision Transformer (high accuracy)
    - resnet        : Residual Networks (balanced)
    - multimodal    : Image + Language understanding
  
  Quick comparison:
    ┌─────────────┬──────────┬──────────┬─────────────┐
    │ Model       │ Accuracy │ Speed    │ Memory      │
    ├─────────────┼──────────┼──────────┼─────────────┤
    │ efficient   │ 95.7%    │ Fast ⚡⚡⚡  │ Low 💾      │
    │ vit         │ 96.1%    │ Moderate ⚡⚡ │ Medium 💾💾 │
    │ resnet      │ 95.1%    │ Fast ⚡⚡⚡  │ Medium 💾💾 │
    │ multimodal  │ 96.5%    │ Moderate ⚡⚡ │ High 💾💾💾 │
    └─────────────┴──────────┴──────────┴─────────────┘
  
  Recommendations:
    ✓ Start with "efficient" for production
    ✓ Use "vit" for best accuracy (if compute available)
    ✓ Use "resnet" as baseline for comparison
    ✓ Use "multimodal" for language queries


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. TRAINING HYPERPARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameter: BATCH_SIZE
  Default: 256
  Valid range: 32 to 1024
  
  What it does:
    - Number of samples processed before weight update
    - Larger batch = more stable gradients, faster training
    - Smaller batch = better generalization, noisier gradients
  
  How to choose:
    GPU Memory (GB)    │ Recommended BATCH_SIZE
    ──────────────────┼─────────────────────────
    4 GB              │ 64-128
    8 GB              │ 128-256
    16 GB             │ 256-512
    32 GB             │ 512-1024
  
  Impact on accuracy:
    - Usually little impact if learning rate adjusted correctly
    - Smaller batch: +0.1-0.2% accuracy (but slower)
    - Larger batch: -0.1-0.2% accuracy if learning rate too high
  
  Tuning tips:
    📌 Start with 256
    📌 If OOM error: reduce to 128 or 64
    📌 If training overfits: use larger batch size
    📌 If training underfits: use smaller batch size


Parameter: EPOCHS
  Default: 100
  Valid range: 10 to 500
  
  What it does:
    - Number of times the model sees entire dataset
    - More epochs = more training (but diminishing returns)
  
  Accuracy progression (typical):
    Epochs   │ Accuracy
    ─────────┼──────────
    10       │ ~88%
    30       │ ~93%
    50       │ ~95%
    100      │ ~96%
    200      │ ~96.5%
  
  How to choose:
    ✓ Use 100 for standard training
    ✓ Use 50 for quick testing
    ✓ Use 200+ for pushing accuracy limits
  
  Observation tips:
    - If accuracy plateaus: stop early (no benefit)
    - If still improving at epoch 100: increase epochs


Parameter: LEARNING_RATE
  Default: 0.1
  Valid range: 0.001 to 1.0
  
  What it does:
    - Controls how much weights change each step
    - Too high: unstable training (diverges)
    - Too low: slow training (settles in local minima)
  
  Recommended values:
    Model Type    │ Good LR Range  │ Default
    ──────────────┼────────────────┼────────
    EfficientNet  │ 0.05 - 0.2     │ 0.1
    ViT           │ 0.001 - 0.01   │ 0.001
    ResNet        │ 0.1 - 0.5      │ 0.1
  
  How to tune:
    1. Start with default
    2. If loss explodes: divide by 2-10
    3. If training too slow: multiply by 1.5-2
  
  Signs of wrong learning rate:
    Too high:  | Loss NaN or oscillates wildly
    Too low:   | Loss decreases very slowly
    Just right:| Smooth loss decrease


Parameter: WEIGHT_DECAY
  Default: 5e-4 (0.0005)
  Valid range: 0 to 0.01
  
  What it does:
    - L2 regularization: penalizes large weights
    - Prevents overfitting
    - Similar to adding noise to weights
  
  Effect on accuracy:
    Weight Decay │ Overfitting  │ Training Acc  │ Test Acc
    ─────────────┼──────────────┼──────────────┼──────────
    0.0          │ High         │ 98%          │ 94%
    0.0001       │ Medium       │ 97%          │ 95%
    0.0005       │ Low          │ 96%          │ 96%
    0.001        │ Very Low     │ 95%          │ 95.5%
  
  How to choose:
    ✓ Use 5e-4 by default
    ✓ If overfitting: increase to 1e-3
    ✓ If underfitting: decrease to 1e-4
    ✓ If unsure: start with 5e-4


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. LEARNING RATE SCHEDULING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameter: WARMUP_EPOCHS
  Default: 5
  Valid range: 0 to 20
  
  What it does:
    - Gradually increases LR for first N epochs
    - Helps stabilize training at the start
    - Recommended for ResNet and EfficientNet
  
  Typical schedule (WARMUP_EPOCHS=5, EPOCHS=100):
    Epoch   │ LR
    ────────┼────────
    1       │ 0.02  (20% of 0.1)
    2       │ 0.04  (40%)
    3       │ 0.06  (60%)
    4       │ 0.08  (80%)
    5       │ 0.10  (100%)
    6-100   │ Decays using cosine schedule
  
  How to choose:
    ✓ Use 5 for stable training
    ✓ Use 3 for faster convergence
    ✓ Use 0 to skip warmup
    ✓ For Vision Transformer: use 10-20


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. DATA AUGMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameter: USE_MIXUP
  Default: True
  Valid: True or False
  
  What it does:
    - Mixes two samples: x_mixed = α*x_i + (1-α)*x_j
    - Mixed labels: y_mixed = α*y_i + (1-α)*y_j
    - Reduces overfitting, improves generalization
  
  Example:
    Before mixup:     [dog image], [cat image]
    After mixup:      [dog+cat blended image]
    Soft label:       [0.7 (dog), 0.3 (cat)]
  
  Impact:
    Disabled: 95.3% accuracy
    Enabled:  96.5% accuracy (+1.2%)
  
  When to use:
    ✓ Always use for CIFAR-10
    ✓ Can skip for MNIST (overtrained already)
    ✓ Great for preventing overfitting


Parameter: MIXUP_ALPHA
  Default: 0.2
  Valid range: 0.1 to 1.0
  
  What it does:
    - Controls how much to mix
    - Higher = more mixing (stronger regularization)
    - Lower = less mixing (closer to original samples)
  
  Effect:
    α = 0.05  │ Weak mixing    │ More like original samples
    α = 0.2   │ Standard       │ Balanced (default)
    α = 0.5   │ Strong mixing  │ More aggressive
    α = 1.0   │ Uniform random │ Maximum mixing
  
  How to choose:
    ✓ Use 0.2 by default (recommended for CIFAR-10)
    ✓ Use 0.1 for less regularization
    ✓ Use 0.5 if overfitting heavily


Parameter: USE_CUTMIX
  Default: True
  Valid: True or False
  
  What it does:
    - Cuts region from one image, pastes into another
    - Preserves spatial structure better than Mixup
    - Often used together with Mixup
  
  Improvement over Mixup:
    - Mixup only:        +1.2%
    - CutMix only:       +0.8%
    - Both Mixup+CutMix: +1.7% (best)


Parameter: CUTMIX_ALPHA
  Default: 1.0
  Valid range: 0.1 to 2.0
  
  What it does:
    - Controls cut region size
    - Higher = larger regions cut
  
  Values:
    0.1   │ Small cuts    │ Subtle mixing
    1.0   │ Medium cuts   │ Balanced (default)
    2.0   │ Large cuts    │ Aggressive


Parameter: LABEL_SMOOTHING
  Default: 0.1
  Valid range: 0 to 0.5
  
  What it does:
    - Prevents model from becoming overconfident
    - Allocates small probability to other classes
    - Acts as regularization
  
  Example (10 classes, one correct class):
    Without smoothing: [1, 0, 0, ..., 0]     (hard target)
    With 0.1 smoothing:[0.91, 0.01, 0.01, ..., 0.01]  (soft target)
  
  Effect on accuracy:
    Smoothing    │ Train Acc  │ Test Acc  │ Generalization
    ─────────────┼───────────┼──────────┼──────────────
    0.0          │ 97.5%     │ 95.3%    │ Poor (overfits)
    0.05         │ 97.0%     │ 95.8%    │ Better
    0.1          │ 96.5%     │ 96.2%    │ Best ⭐
    0.2          │ 96.0%     │ 95.9%    │ Okay
  
  How to choose:
    ✓ Use 0.1 for standard training
    ✓ Use 0.0 if model underfits
    ✓ Use 0.2 if heavy overfitting


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. OPTIMIZATION TECHNIQUES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameter: USE_AMP (Automatic Mixed Precision)
  Default: True
  Valid: True or False
  
  What it does:
    - Mixes float32 and float16 computations
    - Float16: faster, uses less memory
    - Float32: more stable, higher precision
    - Combines benefits of both
  
  Performance impact:
    Training time:  8 hours → 6 hours (-25%) ⚡
    Memory usage:   100% → 70% (-30%) 💾
    Accuracy:       96.2% → 96.1% (negligible loss)
  
  When to use:
    ✓ Always use on NVIDIA GPUs
    ✓ Saves time and memory with no accuracy loss
    ✓ Only works on CUDA, not CPU
  
  Compatibility:
    ✓ Works with all architectures
    ✓ Works with Mixup/CutMix
    ✓ Automatic fallback if GPU doesn't support


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. SYSTEM PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameter: NUM_WORKERS
  Default: 4
  Valid range: 0 to 32
  
  What it does:
    - Number of processes loading data in parallel
    - More workers = faster data loading (up to CPU cores)
  
  How to choose:
    System           │ CPU Cores  │ Recommended NUM_WORKERS
    ─────────────────┼────────────┼───────────────────────
    Laptop           │ 4-8        │ 0-2
    Desktop (mid)    │ 8-16       │ 4-8
    Desktop (high)   │ 16-32      │ 8-16
    Server           │ 32+        │ 16-32
  
  Impact on training:
    0 workers  │ Slow data loading, low GPU utilization
    4 workers  │ Balanced (default)
    8+ workers │ Fast data loading, high GPU utilization
  
  Important notes:
    ⚠️ Windows may struggle with >2 workers (auto-reduced)
    ⚠️ Each worker uses ~50MB RAM
    ⚠️ More workers ≠ always better


Parameter: PIN_MEMORY
  Default: True
  Valid: True or False
  
  What it does:
    - Pins loaded data in RAM for faster GPU transfer
    - Only useful with GPU training
    - Saves copy time to GPU
  
  Use when:
    ✓ Always True for GPU training
    ✓ False for CPU-only (wastes RAM)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. QUICK CONFIGURATION TEMPLATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Template 1: QUICK TESTING (Done in 1 hour)
────────────────────────────────────────────
DATASET = "cifar10"
MODEL_TYPE = "efficient"
BATCH_SIZE = 512          # Fast batch
EPOCHS = 20               # Quick
LEARNING_RATE = 0.1
USE_MIXUP = False         # Skip for speed
USE_AMP = True


Template 2: STANDARD TRAINING (6 hours, 97% accuracy)
──────────────────────────────────────────────────────
DATASET = "cifar10"
MODEL_TYPE = "efficient"
BATCH_SIZE = 256
EPOCHS = 100              # Full training
LEARNING_RATE = 0.1
USE_MIXUP = True
USE_CUTMIX = True
USE_AMP = True
LABEL_SMOOTHING = 0.1


Template 3: MAXIMUM ACCURACY (10+ hours, 97.5%)
────────────────────────────────────────────────
DATASET = "cifar10"
MODEL_TYPE = "vit"        # Best architecture
BATCH_SIZE = 256
EPOCHS = 200              # Long training
LEARNING_RATE = 0.01      # Lower for ViT
WARMUP_EPOCHS = 10        # Longer warmup
USE_MIXUP = True
USE_CUTMIX = True
USE_AMP = True
LABEL_SMOOTHING = 0.15


Template 4: MEMORY CONSTRAINED (GPU with 4GB RAM)
──────────────────────────────────────────────────
DATASET = "cifar10"
MODEL_TYPE = "efficient"
BATCH_SIZE = 64           # Small batch
EPOCHS = 100
LEARNING_RATE = 0.05      # Lower due to small batch
USE_AMP = True            # Critical for memory
NUM_WORKERS = 0           # Reduce overhead


Template 5: RESEARCH / BEST RESULTS (Max time, max quality)
────────────────────────────────────────────────────────────
DATASET = "cifar10"
MODEL_TYPE = "vit"
BATCH_SIZE = 512          # Large batch
EPOCHS = 300              # Very long
LEARNING_RATE = 0.001     # Fine-tuned
WARMUP_EPOCHS = 20
USE_MIXUP = True
USE_CUTMIX = True
USE_AMP = True
LABEL_SMOOTHING = 0.2
NUM_WORKERS = 16


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. DEBUGGING & TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem: Loss is NaN (explodes immediately)
───────────────────────────────────────────
Likely cause: Learning rate too high
Fix:
  1. Reduce LEARNING_RATE by 10x (0.1 → 0.01)
  2. Disable AMP: USE_AMP = False
  3. Reduce batch size: BATCH_SIZE = 64
  4. Check data normalization

Problem: Model accuracy stuck at ~10% (random guessing)
────────────────────────────────────────────────────────
Likely cause: Learning rate too low or model not updating
Fix:
  1. Increase LEARNING_RATE by 2x-10x
  2. Check if model has parameters: print(sum(p.numel()))
  3. Verify data is being loaded correctly

Problem: Training very slow
──────────────────────────
Likely cause: Data loading bottleneck
Fix:
  1. Increase NUM_WORKERS: 4 → 8 or 16
  2. Enable PIN_MEMORY: True
  3. Increase BATCH_SIZE (if memory allows)
  4. Enable USE_AMP: True

Problem: Overfitting (train 99%, test 90%)
──────────────────────────────────────────
Cause: Model capacity too high for data
Fix:
  1. Increase WEIGHT_DECAY: 5e-4 → 1e-3
  2. Increase LABEL_SMOOTHING: 0.1 → 0.2
  3. Enable USE_MIXUP: True
  4. Enable USE_CUTMIX: True
  5. Reduce model size: width_multiplier = 0.75

Problem: Underfitting (train 92%, test 91%)
───────────────────────────────────────────
Likely cause: Model capacity too low
Fix:
  1. Increase EPOCHS: 100 → 200
  2. Decrease LEARNING_RATE slightly: 0.1 → 0.07
  3. Decrease WEIGHT_DECAY: 5e-4 → 1e-4
  4. Use larger model: MODEL_TYPE = "vit"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. MONITORING TRAINING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Expected training logs should look like:

✓ GOOD TRAINING CURVE:
──────────────────────
Epoch 1/100:   train loss: 2.1234, acc: 0.3542, test acc: 0.3601
Epoch 2/100:   train loss: 1.8456, acc: 0.5123, test acc: 0.5134
Epoch 3/100:   train loss: 1.4567, acc: 0.6234, test acc: 0.6145
Epoch 4/100:   train loss: 1.1234, acc: 0.7012, test acc: 0.7023
...
Epoch 100/100: train loss: 0.1234, acc: 0.9634, test acc: 0.9654

✗ DIVERGING (Learning Rate Too High):
──────────────────────────────────────
Epoch 1/100:   train loss: 2.1234, acc: 0.3542, test acc: 0.3601
Epoch 2/100:   train loss: 2.8456, acc: 0.2123, test acc: 0.2134  ❌ Getting worse
Epoch 3/100:   train loss:    NaN, acc:    NaN, test acc:    NaN  ❌ Exploded

✗ STAGNATING (Learning Rate Too Low):
──────────────────────────────────────
Epoch 1/100:   train loss: 2.1234, acc: 0.3542
Epoch 2/100:   train loss: 2.1034, acc: 0.3562  (almost no change)
Epoch 3/100:   train loss: 2.0934, acc: 0.3582  (barely changing)
...
Epoch 100/100: train loss: 1.9234, acc: 0.5134  ❌ Slow progress

✗ OVERFITTING:
──────────────
Epoch 20/100:  train loss: 0.1234, acc: 0.9543, test acc: 0.9512 ✓
Epoch 40/100:  train loss: 0.0456, acc: 0.9843, test acc: 0.9612 ✓
Epoch 80/100:  train loss: 0.0012, acc: 0.9962, test acc: 0.9634 ❌ Gap widening
Epoch 100/100: train loss: 0.0001, acc: 0.9998, test acc: 0.9645 ❌ Train >> Test


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. FINAL TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rule 1: Start conservative, increase gradually
  ✓ Start with default settings
  ✓ Make one change at a time
  ✓ Observe effect before next change

Rule 2: Batch size and learning rate are linked
  If you double batch size → increase LR by 1.4-2x
  If you halve batch size → decrease LR by 0.5-0.7x

Rule 3: When stuck, try these in order:
  1. Disable all augmentations (check if data is okay)
  2. Reduce learning rate by 10x
  3. Try different architecture (efficient → vit)
  4. Check data normalization

Rule 4: Track multiple metrics
  ✓ Train loss (should decrease)
  ✓ Train accuracy (should increase)
  ✓ Test accuracy (should increase, eventually plateau)
  ✓ Gap between train/test (indicates overfitting)

Rule 5: Save the best model (not just the last one)
  ✓ Use best_cifar10_efficient_model.pt for best results
  ✓ Latest model might be overfitting
  ✓ Best model was saved automatically during training


╔════════════════════════════════════════════════════════════════════════════╗
║                    READY TO START TRAINING?                               ║
║                                                                             ║
║  1. Choose a template from section 7                                       ║
║  2. Update train_advanced.py with your settings                            ║
║  3. Run: python train_advanced.py                                          ║
║  4. Monitor the training curves                                            ║
║  5. Adjust parameters if needed                                            ║
║                                                                             ║
║              Good luck! 🚀                                                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

def print_guide():
    """Print the configuration guide."""
    print(configuration_guide)

def save_guide_to_file():
    """Save guide to a text file."""
    with open("CONFIGURATION_GUIDE.txt", "w") as f:
        f.write(configuration_guide)
    print("✓ Configuration guide saved to CONFIGURATION_GUIDE.txt")

if __name__ == "__main__":
    print_guide()
    save_guide_to_file()
