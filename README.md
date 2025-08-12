# Improving Minority-Class Detection in CNN-Based Rice Leaf Diagnosis

This repository contains the official MATLAB code for the research paper titled "Oversampling and Cost-Sensitive SVM for Imbalanced Rice Disease Classification".

## Description
This project investigates the effectiveness of several feature-level class-balancing remedies (SMOTE variants, Cost-Sensitive SVM, and a hybrid approach) applied to the state-of-the-art RiPa-Net pipeline for rice disease diagnosis. The goal is to improve the detection of rare, minority disease classes in the imbalanced Paddy Doctor dataset.

## How to Run the Code
The code is organized into two main stages: baseline replication and the evaluation of the extension remedies. The scripts are located in the `/src/` directory and are designed to be run sequentially.

1.  **Replication:** Run the scripts in `/src/replication/` in order from `phase1` to `phase6`.
2.  **Extension:** Run the scripts in `/src/extension/` to reproduce the results for each of the five balancing remedies.

## Dataset
This study uses the public "Paddy Doctor" dataset, which can be downloaded from Kaggle: [https://www.kaggle.com/competitions/paddy-disease-classification](https://www.kaggle.com/competitions/paddy-disease-classification)

## Dependencies
* MATLAB (Version R2022a or newer)
* Deep Learning Toolbox™
* Statistics and Machine Learning Toolbox™
* Signal Processing Toolbox™
* Wavelet Toolbox™