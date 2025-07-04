# Task A - Face Detection using AMR-CD Model

This repository contains the code and architecture for **Task A** of COMSYS Hackathon 5: **Face Detection** using a novel **AMR-CD (Attention-based Multi-Representation for Cross-Domain)** model. The system performs binary classification to determine if a given face belongs to a known identity, even under grayscale or color variations.

# Model Architecture

<img src="image/taska-1.png" alt="AMR-CD Model" width="400"/>


---

## 🧠 Model Architecture

The AMR-CD model is designed to handle multimodal visual inputs — RGB and grayscale — by combining their strengths through an attention-based fusion.

### ➤ Components:

- **Gray Branch**:  
  - Backbone: `ResNet18`  
  - Input: Grayscale image (1-channel)  
  - Output: 512D → Projected to 256D

- **RGB Branch**:  
  - Backbone: `EfficientNet-B0`  
  - Input: RGB image (3-channel)  
  - Output: 1280D → Projected to 256D

- **Auxiliary Heads**:  
  - One for each branch, used during training to guide each stream independently (via `aux_weight`)

- **Attention-based Fusion**:  
  - Learns a soft attention mask to blend the gray and RGB embeddings  
  - Final embedding: 256D fused representation

- **Classifier**:  
  - Fully connected layers with dropout  
  - Output: Binary label (0 or 1)

---

## 🔬 Methodology

### ✔ Motivation:
Faces may be captured in different modalities — grayscale from old/low-light sources, RGB from standard imaging. Instead of converting or discarding one domain, we fuse both for robust performance.

### ✔ Approach:
1. **Dual Encoders** learn modality-specific features.
2. **Attention Module** weighs which modality contributes more per sample.
3. **Auxiliary Supervision** boosts intermediate learning.
4. **Final Prediction** is made using the fused 256D representation.

---

## ⚙️ Setup

Ensure Python ≥ 3.7 is installed.

### Install Requirements

```bash
pip install -r requirements.txt 
```

Training
To train the AMR-CD model from scratch:

```bash
python main.py \
  --train_dir /path/to/train \
  --val_dir /path/to/val \
  --num_epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --results_file results.csv
  ```

🔍 Inference
To run inference on the test dataset using a trained model:

```bash
python TaskA_test.py \
  --model_path TASK_A_MODEL.pth \
  --test_dir /path/to/test \
  --batch_size 32 \
  --save_predictions \
  --output_file TaskA_test_results.csv
  ```

## This will:

Load the trained AMR-CD model

Evaluate the model on the test set

Print classification metrics (accuracy, precision, recall, F1)

Save per-image predictions to TaskA_test_results.csv

Save a summary of metrics to TaskA_test_results_summary.csv