
# Time Series Self-Supervised Learning (SSL) Notebook

This notebook provides a guide to implementing a self-supervised learning (SSL) approach on time-series data using a convolutional neural network (CNN) in PyTorch. The notebook covers data loading, preprocessing, and training of a neural network for time-series feature extraction and analysis.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training-and-evaluation)
- [Results & Visualization](#results-and-visualization)

## Overview
Self-supervised learning (SSL) allows neural networks to learn features from data without requiring labeled samples. This notebook applies SSL on time-series data (e.g., sensor readings) using a CNN model, providing a framework that can be adapted for various time-series classification tasks.

## Setup

### Requirements
The notebook requires the following packages:
- `torch` (for building the neural network)
- `scikit-learn` (for data preprocessing and metrics)

Install dependencies by running:
```bash
!pip install torch scikit-learn
```

## Data Preparation
1. **Mount Google Drive** (if using Google Colab): This notebook assumes the data is stored on Google Drive.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Load Data**: Data should be in `.pt` format (PyTorch tensors), containing training, validation, and testing datasets.
    ```python
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)
    ```

3. **Data Preprocessing**: The notebook uses `SimpleImputer` to handle missing values and `StandardScaler` to standardize the data. After preprocessing, data is organized into `DataLoaders` for efficient batching during training.

## Model Architecture
The model is a simple convolutional neural network (CNN) defined in PyTorch. The architecture consists of:
- **Convolutional Layers**: Extract features from time-series data.
- **Fully Connected Layers**: Produce final feature embeddings.

Example code snippet:
```python
class SelfSupervisedModel(nn.Module):
    def __init__(self):
        super(SelfSupervisedModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 206, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
```

## Training and Evaluation
1. **Define Training Loop**: The notebook includes a training loop that computes loss and updates model weights.
2. **Metrics**: Accuracy, precision, recall, and F1 score are calculated to evaluate model performance.

## Results and Visualization
After training, the notebook provides visualizations of model performance metrics, such as accuracy and loss over epochs.

---

This notebook serves as a starting point for experimenting with SSL approaches on time-series data and can be modified to include different model architectures or data preprocessing techniques.
