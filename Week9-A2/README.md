# Week 9 - Activity 2: User Knowledge Modeling Dataset

## Overview

This activity analyses the **User Knowledge Modeling Dataset**. The dataset contains student learning behaviour and exam performance values. The goal is to classify the user knowledge level and discover learning patterns using clustering.

The target variable is **UNS**, which represents the user knowledge level:

- `very_low`
- `low`
- `middle`
- `high`


| Feature  | Meaning                                     |
|----------|---------------------------------------------|
| STG      | Study time for goal object materials        |
| SCG      | Repetition number for goal object materials |
| STR      | Study time for related objects              |
| LPR      | Exam performance for related objects        |
| PEG      | Exam performance for goal objects           |


### 1. Load and Explore Data

The Excel file contains separate sheets for training and test data. The project loads both sheets and combines them for exploration. Basic checks are completed for row count, columns, missing values, duplicates, and class distribution.

### 2. Preprocess and Clean Data

The cleaning process includes:
- Removing unused note columns from the Excel file.
- Stripping extra spaces from column names, especially the target column `UNS`.
- Converting feature columns to numeric values.
- Standardising class labels, for example `Very Low` becomes `very_low`.
- Removing invalid rows and duplicate rows.

The cleaned dataset contains **403 records**:

- Training data: **258 records**
- Test data: **145 records**

### 3. Classification Analysis

Three classification models were trained and compared:

| Model         | Accuracy | F1 Score |
|---------------|---------:|---------:|
| Random Forest | 0.9103   | 0.9101   |
| SVM (RBF)     | 0.9034   | 0.9026   |
| KNN (k=5)     | 0.7862   | 0.7762   |

The best model was **Random Forest**, with an accuracy of **91.03%** and F1 score of **91.01%**.

### 4. Clustering Analysis

K-Means clustering was used to group students based on their learning behaviour and performance. The clustering was tested using different values of `k` from 2 to 6.

The best silhouette score was found at **k = 5**, but the project also used **k = 4** for interpretation because the dataset has four known knowledge levels.

### 5. Model Comparison and Interpretation

The classification results show that the dataset can predict user knowledge level with strong performance. The Random Forest model performed best because it can handle non-linear relationships between study behaviour and exam performance.

The clustering results show that natural groups exist, but the clusters do not perfectly match the four knowledge labels. This suggests that knowledge level is not only separated by one factor. It is influenced by a combination of study time, repetition, and exam performance.



### Best Classification Model
**Random Forest** was the best model.
- Accuracy: **91.03%**
- F1 Score: **91.01%**
- It performed slightly better than SVM and clearly better than KNN.

### Important Feature
The feature importance chart shows which variables helped the Random Forest model most. In this dataset, **PEG**, the exam performance for goal objects, is one of the strongest predictors of user knowledge level.

### Class Distribution
The dataset has more records for `low`, `middle`, and `high` knowledge levels than `very_low`. This is important because class imbalance can affect model learning and evaluation.

### Clustering Insight
K-Means helped show groups of students with similar learning behaviour. However, clustering was weaker than supervised classification because it does not use the known knowledge labels during training.


## Key Insights
1. The dataset is suitable for classification because the target knowledge level is already labelled.
2. Random Forest gave the best result, so it is the recommended model for this activity.
3. Goal object exam performance is a strong factor in predicting knowledge level.
4. Clustering is useful for discovering groups, but it is less accurate than classification for this dataset.
5. Students with similar scores and study behaviours can still belong to different knowledge levels, so multiple features should be considered together.

## Conclusion
This project shows a complete data analysis process using the User Knowledge Modeling Dataset. The data was cleaned, explored, classified, and clustered. The best result came from the Random Forest classifier, which achieved about **91% accuracy**. The analysis also showed that exam performance for goal objects is a key predictor of user knowledge level.
