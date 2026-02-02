# CLAUDE.md - Pix2Pix-Flow Codebase Guide

## Project Overview

Pix2Pix-Flow is an image-to-image translation system using **flow-based generative models** (normalizing flows). It combines the pix2pix framework with OpenAI's Glow architecture to learn invertible mappings between paired image domains (Domain A <-> Domain B). The approach trains two separate flow models whose latent codes are aligned via a shared loss, enabling unsupervised cross-domain translation.

**Supported datasets:** MNIST, CIFAR-10, Edges2Shoes, CelebA, LSUN, ImageNet

## Repository Structure

```
pix2pix-flow/
├── model.py              # Core flow model: encoder/decoder, RevNet2D, coupling layers, loss
├── train.py              # Training loop with Horovod distributed training, checkpointing
├── tfops.py              # TF ops: convolutions, actnorm normalization, activations
├── optim.py              # Optimizers (Adam/Adamax) and Polyak EMA averaging
├── eval.py               # MSE evaluation between latent codes
├── graphics.py           # Image grid visualization (numpy -> PNG)
├── utils.py              # JSON result logging, npy-to-image conversion
├── memory_saving_gradients.py  # Gradient checkpointing for memory efficiency
├── requirements.txt      # Python dependencies
│
├── data_loaders/         # Dataset loading modules
│   ├── get_data.py                # TFRecord-based data pipeline
│   ├── get_mnist_cifar_joint.py   # MNIST/CIFAR10 paired loading via Keras
│   ├── get_edges_shoes_joint.py   # Edges2Shoes .npy loading
│   └── generate_tfr/             # TFRecord generation utilities
│
└── demo/                 # Interactive web demo (Flask + JS)
    ├── model.py          # Pre-trained model inference
    ├── server.py         # Flask server (port 5000)
    ├── align_face.py     # Face alignment for CelebA
    ├── get_manipulators.py  # Latent manipulation vectors
    ├── script.sh         # Setup: downloads weights, dlib model
    └── web/              # Frontend (HTML/JS/CSS)
```

## Architecture

Two independent flow models (Model A, Model B) each implement a hierarchical multi-scale invertible architecture:

1. **Squeeze2D** - reduces spatial dims, increases channels
2. **RevNet2D** - stack of invertible blocks (actnorm -> permutation -> coupling)
3. **Split2D** - multi-scale decomposition storing intermediate latent codes
4. **Prior** - learned Gaussian spatial prior (optionally class-conditioned)

Flow permutation types: reverse (0), shuffle (1), invertible 1x1 conv (2).
Coupling types: additive (0), affine (1).

**Joint training loss:**
```
Loss_A = MLE_loss_A + code_loss_scale * ||code_A - code_B||^2
Loss_B = MLE_loss_B + code_loss_scale * ||code_A - code_B||^2
```

## Dependencies

- **TensorFlow 1.8** (GPU, uses tf.contrib extensively)
- **Horovod 0.13.8** (distributed training via MPI/allreduce)
- **Keras 2.2.0** (dataset loading only)
- **Pillow 5.2.0**, **toposort 1.5**, NumPy, SciPy

## Common Commands

### Training (multi-GPU)
```bash
mpiexec -n 4 python train.py \
  --problem edges2shoes \
  --image_size 32 \
  --n_level 3 \
  --depth 32 \
  --flow_permutation 2 \
  --flow_coupling 1 \
  --seed 0 \
  --learntop \
  --lr 0.001 \
  --n_bits_x 8 \
  --joint_train \
  --logdir ./logs
```

### Training (single GPU, debug)
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --depth 1 \
  --epochs_full_sample 1 \
  --epochs_full_valid 1 \
  --joint_train \
  --problem mnist \
  --logdir ./logs-debug
```

### Inference (encode/decode test set)
```bash
python train.py \
  --problem edges2shoes \
  --image_size 32 --n_level 3 --depth 32 \
  --flow_permutation 2 --flow_coupling 1 \
  --seed 0 --learntop --lr 0.001 --n_bits_x 8 \
  --joint_train --logdir LOGDIR --inference
```
Produces: `z_A.npy`, `z_B.npy` (latent codes), `A2B.png`, `B2A.png` (translation grids)

### Evaluation
```bash
python eval.py --A z_A.npy --B z_B.npy
```

### Demo server
```bash
cd demo && ./script.sh        # one-time setup
python server.py              # Flask API on :5000
python -m http.server 8000    # static UI on :8000, open 0.0.0.0:8000/web
```

## Key Configuration Parameters

All configuration is via command-line arguments to `train.py` (no config files).

| Category | Flags | Notes |
|----------|-------|-------|
| Model | `--depth`, `--width`, `--n_levels`, `--image_size`, `--n_bits_x` | Depth = RevNet blocks per level |
| Flow | `--flow_permutation` (0/1/2), `--flow_coupling` (0/1), `--learntop` | Type 2 perm + affine coupling recommended |
| Optimizer | `--lr`, `--optimizer` (adam/adamax), `--beta1`, `--weight_decay`, `--epochs_warmup` | LR warmup is linear |
| Data | `--problem`, `--data_dir`, `--dal` (augmentation level), `--n_batch_train` | dal=0 means no augmentation |
| Pix2Pix | `--joint_train`, `--code_loss_type`, `--code_loss_scale`, `--mle_loss_scale` | joint_train enables dual-model training |
| Checkpoint | `--logdir`, `--restore_path_A`, `--restore_path_B` | Best model saved as `model_{A,B}_best_loss.ckpt` |

## Output Artifacts

Training produces in `--logdir`:
- `model_A_best_loss.ckpt`, `model_B_best_loss.ckpt` - best checkpoints
- `train_A.txt`, `train_B.txt` - training logs (JSON lines)
- `test_A.txt`, `test_B.txt` - validation logs
- `*_epoch_*_sample_*.png` - generated sample images

## Code Conventions

- **Functional style**: minimal OOP, heavy use of nested functions and closures
- **Naming**: lowercase_with_underscores; `_A`/`_B` suffixes for domain-specific tensors; `z` for latent, `x` for images, `eps` for noise, `hps` for hyperparameters
- **Imports**: `import tfops as Z` is used throughout; data loaders imported conditionally based on `--problem`
- **TF patterns**: `tf.variable_scope` for namespacing, `@add_arg_scope` decorators, `hvd.rank() == 0` guards for master-only operations
- **Logging**: only rank 0 prints/logs; JSON-line format via `ResultLogger`
- **No test suite**: no pytest/unittest infrastructure exists; validation happens during training

## Development Notes

- The codebase targets **TensorFlow 1.x** and uses `tf.contrib` heavily (not compatible with TF 2.x without migration)
- Distributed training requires MPI (`mpiexec`/`mpirun`) and Horovod
- `memory_saving_gradients.py` patches TF's gradient computation to reduce VRAM usage - imported and activated in `train.py`
- The `DEBUG` flag in `train.py` is set to `True` by default for activation statistics printing
- Data is expected in `--data_dir` with dataset-specific subdirectory structure (e.g., `edges2shoes_32/train/A.npy`)
- `.gitignore` excludes `logs/`, `logs-*/`, `tmp/`, `*.npy` - never commit model weights or large data files
