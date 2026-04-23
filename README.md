# Leakage-Free Machine Learning Framework for Soft Clay Stabilization

## Overview
This repository contains the dataset, source code, and pre-trained models for the research article: *"A Leakage-Free Machine Learning Framework for Predicting Continuous Stress-Strain Behavior of Alkali-Stabilized Soft Clay."* The primary objective is to address the "data leakage" problem in geotechnical machine learning by using GroupKFold cross-validation and polynomial feature engineering ($\epsilon^2$ and $\epsilon^3$). The proposed Artificial Neural Network (ANN) accurately maps non-linear stress-strain evolution.

## Repository Contents
* **`ANN_Ready_Data.xlsx`**: The experimental dataset (>3000 continuous stress-strain data points).
* **`four_models.py`**: The main Python script containing data preprocessing, feature engineering, and the training architecture for all scenarios. *(Note: Do not use spaces in file names).*
* **`shap_analysis.py`**: The script utilized for Explainable AI (SHAP) analysis.
* **`quick_test.py`**: A lightweight script to quickly verify that the models and environment are functioning correctly.
* **`Model_2_Base_ANN.pkl`**: Pre-trained standard ANN.
* **`Model_3_Ultimate_ANN.pkl`**: Pre-trained Ultimate ANN (The Proposed Model).
* **`Model_4_DataLeakage_ANN.pkl`**: Pre-trained intentionally flawed model.

## Requirements
* Python 3.8+
* `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `openpyxl`

## How to Run the Quick-Test (Example)
To quickly verify the repository without retraining all models, we have provided a quick-test script. This script loads the first 5 rows of the dataset and runs predictions using the pre-trained Ultimate ANN model.
1. Ensure all requirements are installed (`pip install pandas numpy scikit-learn openpyxl`).
2. Open your terminal or command prompt in the repository folder.
3. Run the following command:
   ```bash
   python quick_test.py
