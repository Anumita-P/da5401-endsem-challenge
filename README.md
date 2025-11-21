Endsem Challenge

 Anumita (BE22B004)

## Overview

This project builds a regression model to predict  scores (0–10) using multilingual conversational embeddings. The pipeline uses feature-engineered embeddings combined with a LightGBM regressor to capture semantic relationships between metric definitions, prompts, and responses.

A secondary approach using a 1D CNN was also implemented and compared.

## Dataset Description

Each data sample includes four 768-dimensional embeddings:
- Metric definition
- System prompt
- User prompt
- Model response

The dataset is multilingual (primarily English, Hindi, Bengali, Tamil). Scores are highly imbalanced, with most values between 8 and 10.

### Key Observations
- Strong imbalance in lower scores required augmentation
- High variability in text length (up to 12,000+ characters)
- Metric embeddings form meaningful clusters in PCA space
- Cosine similarity alone does not strongly correlate with scores

## Methodology

### 1. Data Stratification

Since scores are continuous, they were bucketed into coarse bins (low, medium, high) to enable Stratified K-Fold cross-validation.

### 2. Augmentation Strategies

| Strategy | Description | Purpose |
|----------|------------|---------|
| Synthetic negatives | Random mismatched embedding pairs assigned scores 0–0.5 | Improve low-score learning |
| Perfection boosting | Duplicated score-10 samples with noise | Stabilize learning for high scores |

Augmentation was applied only to training folds.

### 3. Feature Engineering

| Feature Type | Description | Motivation |
|--------------|------------|------------|
| Global PCA features | PCA of raw embeddings | Reduce dimension, remove noise |
| Difference vectors | Element-wise difference between embeddings | Capture deviations |
| Element-wise product | Multiplicative interactions | Model interaction effects |
| Cosine similarity + L1/L2 distances | Between metric, instruction, and communication embeddings | Direct semantic alignment signals |

All embeddings were normalized before deriving interaction features.

## Modeling Approach

### Primary Model: LightGBM Regressor

Hyperparameters:
n_estimators = 6000
learning_rate = 0.01
num_leaves = 40
feature_fraction = 0.6
bagging_fraction = 0.7


Training setup:
- Stratified K-Fold (5 folds)
- Early stopping (patience = 200)
- Augmentation only in training folds

Postprocessing:
- Scores clipped to [0, 10]
- Final predictions rounded for submission

Results:
- Leaderboard RMSE: 2.451
- CV performance improved significantly after feature engineering

### Alternative Model: 1D CNN

- Log-scaled targets
- Weighted MAE loss emphasizing low score samples
- Convolutional layers with batch normalization and max pooling
- Global max pooling + regression layer

This method was slower to train and underperformed compared to LightGBM, although it modeled embeddings directly without engineered features.

## Final Pipeline Diagram
→ Load Embeddings + Scores
→ Create Stratification Bins
→ Augment Training Data (Synthetic Negatives, Perfect Score Boosting)
→ Feature Engineering (PCA + Similarity + Interaction Features)
→ LightGBM Regression (Stratified K-Fold + Early Stopping)
→ Evaluate (RMSE)
→ Clip & Round Predictions


## Conclusion

The LightGBM pipeline performed well due to
- Rich, targeted feature engineering
- Handling of score imbalance
- Efficient training and better generalization

