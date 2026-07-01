# VLD — MGDCC Pop-Noise Detection (CNN Classifier)

A 2D-CNN based binary classifier that detects the presence of **pop noise** in speech recordings using **MGDCC (Modified Group Delay Cepstral Coefficients)** features. Built for dysarthric speech research, where clean feature extraction is critical for downstream analysis.

---

## Project Overview

The goal of this project is to classify speech recordings into two classes based on their MGDCC features:

- **With pop noise**
- **Without pop noise**

This is framed as a **binary classification problem**, solved using a custom Convolutional Neural Network trained on pre-extracted MGDCC feature matrices (`.mat` files).

### Pipeline

1. **Data Loading**
   - MGDCC features are loaded from Google Drive as `.mat` files using `scipy.io.loadmat`.
   - Two classes are loaded from separate directories: recordings *with* pop noise and recordings *without* pop noise.

2. **Preprocessing**
   - Each feature matrix has shape `(20, variable_length)`. Since recordings vary in duration, all matrices are **zero-padded** to a fixed width (`208`) so they can be batched.
   - Labels are one-hot/binary encoded using `sklearn.preprocessing.LabelBinarizer`.
   - Final input tensors are reshaped to `(N, 20, 208, 1)` — ready for 2D convolution.

3. **Train / Validation Split**
   - An 80/20 split is performed using `train_test_split` (with a fixed random seed for reproducibility).

4. **Model Architecture**
   A sequential CNN with 5 convolutional blocks:
   - **Conv2D → BatchNorm → ReLU → MaxPool → Dropout**, with filter sizes increasing across blocks: `16 → 32 → 64 → 128 → 256`
   - Followed by a **Flatten** layer and a fully connected head: `Dense(128) → Dense(64) → Dense(16) → Dense(1, sigmoid)`
   - **Loss:** Binary Crossentropy
   - **Optimizer:** Adam (`lr = 0.001`)

5. **Training & Evaluation**
   - Model is trained using **Stratified Shuffle Split (5-fold)** cross-validation to get a robust estimate of performance across different train/test splits.
   - Final model performance is reported via `model.evaluate()` on the held-out validation set.

---

##  Getting Started

### Prerequisites
This notebook was built to run on **Google Colab** with data stored on **Google Drive**.

```bash
pip install tensorflow scikit-learn scipy numpy
```

### Data Setup
The notebook expects two directories of `.mat` MGDCC feature files (mounted via Google Drive):
```
/content/drive/MyDrive/rpa_mgd/save rc a train   # recordings WITH pop noise
/content/drive/MyDrive/rpa_mgd/save rp a train   # recordings WITHOUT pop noise
```

Each `.mat` file should contain a `feat` key holding the MGDCC feature matrix for that recording.

### Running the Notebook
1. Mount your Google Drive when prompted (first cell).
2. Update the `path1` / `path2` variables to point to your own MGDCC feature directories if different.
3. Run all cells sequentially — data loading → padding → train/val split → model definition → K-Fold training → evaluation.









