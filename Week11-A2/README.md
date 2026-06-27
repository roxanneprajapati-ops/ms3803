# Week 11 – Activity 2: New Zealand Wellbeing Statistics (2014–2018) Forecasting Project

## Project Overview

This project develops a simple time-series forecasting model using the **New Zealand Wellbeing Statistics (2014–2018)** dataset from **Stats NZ**. The objective is to predict the next year's wellbeing value by comparing five forecasting techniques:

- Linear Regression
- XGBoost
- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)
- ARIMA

The models are evaluated using common forecasting metrics to identify the best-performing approach.

---

# Dataset

**Source:** Stats NZ – Wellbeing Time Series Explorer

**Selected Measure:**

- Overall Life Satisfaction
- Total Population

The dataset contains historical wellbeing values for 2014, 2016, and 2018.

---

# Data Analytics Process

## 1. Load the Dataset

The Python program reads the Excel workbook and extracts the **Total Population** records from the **Overall Life Satisfaction** worksheet.

---

## 2. Data Preparation and Processing

The dataset is cleaned and prepared before modelling.

The following steps are performed:

- Remove empty records
- Convert Year and wellbeing values to numeric format
- Sort the data by year
- Estimate missing years (2015 and 2017) using linear interpolation
- Create a **Time_Index** feature to represent the time sequence
- Create a **Lag_1** feature using the previous year's wellbeing value

The processed dataset is then saved as a CSV file.

---

## 3. Train-Test Split

The processed data is divided into:

- **Training Data:** Earlier years
- **Testing Data:** Final year (2018)

The training data is used to build the forecasting models, while the testing data is used to evaluate prediction accuracy.

---

## 4. Forecasting Models

Five forecasting techniques are implemented and compared:

- Linear Regression
- XGBoost
- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)
- ARIMA

Each model predicts the wellbeing value for the testing year.

---

## 5. Model Evaluation

The models are evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **MAPE (Mean Absolute Percentage Error)**

The model with the **lowest RMSE** is selected as the best-performing forecasting model.

---

## 6. Next-Year Forecast

After selecting the best-performing model, the program predicts the wellbeing value for the following year (2019).

---

# Generated Outputs

## Results

- Processed dataset
- Model performance comparison
- Model predictions
- Next-year forecast

---

# Top 3 Findings

1. The selected wellbeing measure remained relatively stable between 2014 and 2018.
2. Linear Regression produced the lowest prediction error and was identified as the best-performing forecasting model for this dataset.
3. The forecast indicates that New Zealand's overall life satisfaction is expected to remain relatively stable in the following year.

