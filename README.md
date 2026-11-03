# CIFAR-10 CNN Classifier

A small convolutional neural network for CIFAR-10, built with PyTorch and trained in a single Jupyter notebook.

## Results

**Test accuracy: 66.81%** (6,681 / 10,000)

`best_model.pth` is the checkpoint with the highest validation accuracy during training. The test set was used only for the final evaluation.

## Model

| Layer | Output |
| --- | --- |
| Conv2d(3, 12, 5) + ReLU + MaxPool(2, 2) | 12 × 14 × 14 |
| Conv2d(12, 24, 5) + ReLU + MaxPool(2, 2) | 24 × 5 × 5 |
| Flatten | 600 |
| Linear(600, 120) + ReLU | 120 |
| Linear(120, 84) + ReLU | 84 |
| Linear(84, 10) | 10 |

91,270 trainable parameters.

## Training

- 45,000 training images and 5,000 validation images from the CIFAR-10 training set (seed 42)
- Per-channel normalization with mean and standard deviation of 0.5
- Batch size 128
- SGD with learning rate 0.001 and momentum 0.9
- Cross-entropy loss
- 50 epochs

## Files

- `cnn_cifar10.ipynb` — data loading, model definition, training, evaluation, and plots
- `best_model.pth` — trained weights from the best validation epoch
- `data/` — CIFAR-10 download (not tracked by git)

## Running

Install PyTorch, torchvision, scikit-learn, matplotlib, seaborn, and Jupyter, then run `cnn_cifar10.ipynb`. The dataset is downloaded to `data/` on the first run.
