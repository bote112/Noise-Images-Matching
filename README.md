# Statistical Similarity Learning for Noise Image Classification

## Overview
This project addresses a binary classification problem: determining whether two noise images originate from the same underlying stochastic process.

Unlike traditional image classification tasks that rely on semantic content, this approach treats each image as a realization of a random field. The task is framed as a statistical similarity problem, where classification is based on comparing distributional, spectral, and spatial characteristics.

## Key Idea
Two images generated from the same noise distribution can look completely different at the pixel level. Direct pixel-wise comparison is therefore ineffective. 

Instead, this project:
* Extracts statistical descriptors from each image
* Builds pairwise similarity features between images
* Trains machine learning models to distinguish matching vs. non-matching pairs

This effectively turns the task into a **learned two-sample statistical test**.

## Feature Engineering
Each image is represented using a 17-dimensional feature vector composed of:

### Global Statistical Features
* Mean, standard deviation, variance
* Minimum and maximum values
* Skewness and kurtosis

### Histogram-Based Features
* Entropy (measure of randomness)
* Histogram variance

### Frequency-Domain Features
* Mean and variance of FFT magnitude
* Low-frequency vs. high-frequency energy

### Spatial Features
* Patch-based statistics (local mean and variance behavior)

### Pairwise Feature Construction
For each pair of images, features are combined into a **38-dimensional vector**:
* Absolute differences between features
* Squared differences between features
* Cosine similarity
* Euclidean distance
* Wasserstein distance (distribution comparison)
* Pixel-level Pearson correlation

## Models
Two ensemble-based models were evaluated:

* **Random Forest:** A robust baseline model that handles heterogeneous features well and is resistant to overfitting.
* **XGBoost:** A gradient boosting framework that captures complex feature interactions and includes built-in regularization.

## Results

| Model | Accuracy | MAE | MSE | Spearman | Kendall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 0.784 | 0.284 | 0.152 | 0.629 | 0.514 |
| **Random Forest** | 0.772 | 0.353 | 0.160 | 0.620 | 0.507 |

## Implementation Details
* Feature extraction pipeline with modular design (`extract_features`)
* Efficient feature caching to avoid recomputation
* Standardization using `StandardScaler`
* Stratified 5-fold cross-validation
* Final retraining on combined training and validation data

## Key Takeaways
* Noise comparison is better approached as a statistical problem, not a visual one.
* Handcrafted features can be highly effective when domain knowledge is used.
* Spectral and distributional properties are critical for distinguishing stochastic processes.
