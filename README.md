# Character Recognition in Different Scripts

## Overview
This Jupyter Notebook implements a one-shot learning approach using metric learning for character recognition across different alphabets. The goal is to generalize well to unseen scripts with minimal labeled examples. The model is designed to handle challenges such as unseen character classes, rotated characters, and contextual dependencies.

## Problem Statement
The task is to classify characters from new alphabets using a few annotated examples. Additionally, the model must generalize to rotated characters and incorporate domain knowledge for improved classification accuracy.

## Methods
### 1. Character Recognition
- One-shot learning with metric learning to compare character embeddings.
- `Top-k` accuracy is used to evaluate performance.

### 2. Rotated Character Recognition
- Data augmentation with random rotations.
- Rotation-invariant embedding space using a modified loss function.

### 3. Exploiting Domain Knowledge
- Character sequence probabilities refine predictions.
- Adjusts model outputs based on likely character transitions.

## Implementation Details
### **Libraries Used**
- `numpy`, `torch`, `matplotlib.pyplot`, `pickle`.

### **Main Workflow**
1. **Data Loading**
   - Uses a function to read character datasets from a pickle file.
   - Displays example images and structures the dataset.

2. **Model Development**
   - Uses a deep learning model based on metric learning.
   - Embeddings compare similarity between labeled and unlabeled characters.

3. **Evaluation Metrics**
   - Computes `top-k` accuracy to assess performance.

4. **Handling Rotations**
   - Data augmentation with random rotations.
   - Enforces rotational consistency in embedding space.

5. **Domain Knowledge Integration**
   - Uses character sequence probabilities to refine predictions.

## How to Run
1. Install dependencies (`pip install -r requirements.txt` if applicable).
2. Open the Jupyter Notebook and run the cells sequentially.
3. Train the model using the provided training data.
4. Evaluate the model on test datasets.

## Results
The model effectively generalizes to new alphabets, handles rotations using augmentation and embedding consistency, and improves accuracy with domain knowledge.
