# Week 6 Activity – SVM Classification using Wine Dataset

## Project Description

This activity demonstrates how to train and test a Support Vector Machine (SVM) model using the Wine dataset.

The program:
- Loads the dataset from the `wine/` folder
- Cleans and preprocesses the data
- Splits the dataset into training and testing datasets
- Trains an SVM classifier using a linear kernel
- Evaluates the model performance using different evaluation metrics

---

# Dataset

The dataset used is the Wine dataset.

Dataset file location:

```plaintext
wine/wine.data
```

The dataset contains:
- 178 wine samples
- 13 chemical features
- 3 wine classes

---

# Features Used

The dataset contains the following features:

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315
- Proline

Target column:
- Class

---

# Data Cleaning

The following preprocessing steps were applied:

- Removed duplicate rows
- Checked missing values
- Removed null values
---

# Train and Test Split

The dataset was divided using:

```python
test_size=0.2
```

This means:
- 80% of the dataset is used for training
- 20% of the dataset is used for testing

Dataset split result:
- Training dataset: 142 rows
- Testing dataset: 36 rows

The `random_state=42` was used so the result stays consistent every run.

---

# SVM Model

The model uses:

```python
SVC(kernel="linear")
```

A linear kernel was used to classify the wine classes.

The dataset was also scaled using `StandardScaler()` because the wine features have different numeric ranges.

---

# Evaluation Metrics

The following evaluation metrics were used:

- Accuracy Score
- Classification Report
- Confusion Matrix

---

# Result

## Accuracy Score

```plaintext
94.44%
```

The SVM model correctly predicted most wine classes from the testing dataset.

---

# Classification Report Result Explanation

### Wine Class 1
- Precision: 0.92
- Recall: 1.00
- F1-score: 0.96

The model correctly identified all Wine Class 1 samples.

### Wine Class 2
- Precision: 0.93
- Recall: 0.93
- F1-score: 0.93

The model performed very well for Wine Class 2 with only a small number of incorrect predictions.

### Wine Class 3
- Precision: 1.00
- Recall: 0.90
- F1-score: 0.95

The model correctly predicted most Wine Class 3 samples with very few errors.

---

# Confusion Matrix

```plaintext
[[12  0  0]
 [ 1 13  0]
 [ 0  1  9]]
```

This means:
- Most wine samples were classified correctly
- Only 2 predictions were incorrect

The program displays:
- Dataset information
- Training and testing sizes
- Accuracy score
- Classification report
- Confusion matrix