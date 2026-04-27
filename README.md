# CNN–BiLSTM–Attention for Cross-Project Defect Prediction

## Overview
This project implements a hybrid deep learning model for software defect prediction using a combination of Convolutional Neural Networks (CNN), Bidirectional LSTM (BiLSTM), and Attention mechanisms. The model is designed for Cross-Project Defect Prediction (CPDP), where it is trained on multiple source projects and evaluated on an unseen target project.

---

## Methodology

### 1. Data Handling
- ARFF datasets are loaded using `scipy.io.arff`
- Byte data is decoded into readable format
- Class labels are standardized into binary values (0: non-defective, 1: defective)
- Data is stored in structured Pandas DataFrames

### 2. Preprocessing
- Features and labels are separated
- Standardization is applied using `StandardScaler`
- Data is reshaped into 3D format for deep learning models

### 3. Handling Class Imbalance
- Class weights are computed to balance defective and non-defective samples
- Focal Loss is used to emphasize hard-to-classify samples and reduce bias toward majority class

### 4. Model Architecture
- Multi-scale CNN with kernel sizes 3, 5, and 7 for feature extraction
- Residual connection to preserve original input information
- BiLSTM layer to capture relationships among features
- Multi-head attention to focus on important features
- Global average pooling followed by dense layers
- Sigmoid output layer for binary classification

### 5. Training Strategy
- Optimizer: Adam (learning rate = 0.0005)
- Validation split: 10%
- Early stopping based on validation loss
- Class weights applied during training

### 6. Cross-Project Evaluation
- Train on all projects except one
- Test on the remaining project
- Repeat for all datasets (leave-one-project-out strategy)

### 7. Threshold Optimization
- Prediction probabilities are converted to labels using optimal threshold
- Threshold selected from 0.1 to 0.9 based on best F1-score

### 8. Evaluation Metrics
- F1-score
- Matthews Correlation Coefficient (MCC)
- Area Under ROC Curve (AUC)

---

## Datasets
The model supports ARFF datasets such as:
- Apache
- Safe
- Zxing
- AEEEM datasets (LC, ML, PDE, EQ, JDT)

---

## Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- SciPy

---

## How to Run
1. Place ARFF dataset files in the project directory
2. Update file names in the script if needed
3. Run the script:
   ```bash
   python cnn_bilstm_attention.py
