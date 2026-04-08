# Leakage-Free ML Framework for Soft Clay Stabilization

## Overview
This repository contains the dataset and pre-trained models for the research article: **"A Leakage-Free Machine Learning Framework for Predicting Continuous Stress-Strain Behavior of Alkali-Stabilized Soft Clay."** The primary objective of this project is to address the severe "data leakage" problem commonly caused by random row-wise data splitting in geotechnical machine learning studies. By implementing a sample-based cross-validation strategy (`GroupKFold`) and engineering physical polynomial features (ε² and ε³), the proposed Artificial Neural Network (ANN) successfully maps the non-linear, continuous stress-strain evolution and post-peak softening behavior of NaOH- and KOH-stabilized clays.

## Repository Contents

* **`ANN_Ready_Data.xlsx`**: The complete experimental dataset containing over 3000 continuous stress-strain data points. Features include chemical dosages, curing time, water content, bulk density, and polynomial strain attributes.
* **Pre-trained Models**: 
  * `Model_1_RandomForest.zip`: The tree-based ensemble benchmark model (compressed).
  * `Model_2_Base_ANN.pkl`: The standard ANN architecture evaluated under leakage-free conditions.
  * `Model_3_Ultimate_ANN.pkl`: The optimized ANN architecture utilizing the complete polynomial feature engineering framework (The Proposed Model).
  * `Model_4_DataLeakage_ANN.pkl`: The ANN model evaluated using a conventional random row-wise split, intentionally provided to demonstrate the artificial inflation caused by data leakage.

## Methodology Highlights
1. **Elimination of Data Leakage:** Replaced standard random splitting with `GroupKFold` to ensure independent physical specimens do not overlap between training and testing sets.
2. **Physical Interpretability:** Incorporated quadratic and cubic strain features to constrain the network physically, enabling accurate predictions of brittle collapse and ductile softening.
3. **Explainable AI:** SHAP analysis confirms that the model dynamically activates higher-order features specifically during the critical post-peak degradation phase.

## Usage
You can directly load the provided `.pkl` and `.zip` model files to evaluate model performance without retraining. Ensure that the required data structures match the columns provided in `ANN_Ready_Data.xlsx`.

## Citation
If you utilize this dataset or framework in your research, please consider citing our paper (Citation details will be updated upon publication).
