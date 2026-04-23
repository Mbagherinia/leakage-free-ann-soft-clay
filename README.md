# A Leakage-Free Machine Learning Framework for Predicting Continuous Stress-Strain Behavior of Alkali-Stabilized Soft Clay

## Overview
This repository contains the dataset and source code for the research article: *"A Leakage-Free Machine Learning Framework for Predicting Continuous Stress-Strain Behavior of Alkali-Stabilized Soft Clay."* The primary objective is to address the severe "data leakage" problem commonly caused by random row-wise data splitting in geotechnical machine learning studies. By implementing a sample-based cross-validation strategy (GroupKFold) and engineering physical polynomial features ($\epsilon^2$ and $\epsilon^3$), the proposed Artificial Neural Network (ANN) successfully maps the non-linear, continuous stress-strain evolution and post-peak softening behavior of NaOH- and KOH-stabilized clays.

## Repository Contents
* **`ANN_Ready_Data.xlsx`**: The complete experimental dataset containing over 3000 continuous stress-strain data points. Features include chemical dosages, curing time, water content, bulk density, and polynomial strain attributes.
* **`four_models.py`**: The main Python script containing data preprocessing, polynomial feature engineering, the GroupKFold validation setup, and the training/evaluation architectures for all predictive scenarios (Random Forest Benchmark, Base ANN, Ultimate ANN, and the Data Leakage simulation).
* **`shap_analysis.py`**: The script utilized for Explainable AI (SHAP) analysis to interpret how the network transitions its decision logic during the physical failure and softening phases.
* **`quick_test.py`**: A lightweight script to quickly verify that the coding environment and pipeline execute without errors.

## Requirements
* Python 3.8+
* `pandas`, `numpy`, `scikit-learn`, `shap`, `matplotlib`, `openpyxl`

## How to Run the Quick-Test (Example)
To strictly comply with open-source sharing policies while keeping the repository lightweight, we have omitted large pre-trained model files (e.g., .pkl, .zip). Instead, to quickly verify the repository setup without retraining the heavy models from scratch, we have provided a technical validation script (`quick_test.py`). 

This script loads a tiny subset of the dataset (first 200 rows) and trains a miniature version of the Ultimate ANN framework in less than a second.

1. Ensure all requirements are installed:
   ```bash
   pip install pandas numpy scikit-learn openpyxl
