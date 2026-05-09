## Week 4 - Activity 2: Data Aggregation

# World Happiness SQL Analysis

## Project Overview

This project uses the World Happiness dataset to answer two SQL analysis tasks:

1. Create GDP categories, calculate average happiness per category, and rank countries within each category.
2. Split countries into high and low corruption perception groups, compute multiple averages, and compare the groups using a subquery.

## Dataset Overview

The dataset contains country-level happiness data.

- Records before SQL analysis: {len(df)}
- Number of columns: {len(df.columns)}
- Main columns used:
  - Country
  - Happiness_Score
  - GDP_per_Capita
  - Social_Support
  - Healthy_Life_Expectancy
  - Freedom_to_Make_Choices
  - Generosity
  - Perceptions_of_Corruption

## Query 1: GDP Categories and Ranking

### Purpose

This query groups countries into Low, Medium, and High GDP categories. It then calculates the average happiness score for each GDP category and ranks countries inside each category based on their happiness score.

### Logic

- `CASE` is used to create GDP categories:
  - Low GDP: GDP_per_Capita less than 0.80
  - Medium GDP: GDP_per_Capita from 0.80 to 1.30
  - High GDP: GDP_per_Capita above 1.30
- `AVG(Happiness_Score)` calculates the average happiness per GDP category.
- `RANK()` ranks countries inside each GDP category from highest happiness to lowest happiness.
- A `JOIN` is used to combine country ranking with the category average.

### Result Summary

The High GDP group has the highest average happiness score in this dataset. However, some countries with lower GDP still have competitive happiness scores, showing that happiness is not only affected by GDP.

## Query 2: Corruption Perception Comparison

### Purpose

This query compares countries with high corruption perception and low corruption perception.

### Logic

- A subquery calculates the overall average corruption perception.
- `CASE` splits countries into two groups:
  - High Corruption Perception: corruption score is greater than or equal to the dataset average.
  - Low Corruption Perception: corruption score is below the dataset average.
- The query calculates average values for:
  - Happiness score
  - GDP per capita
  - Social support
  - Healthy life expectancy
  - Freedom
  - Generosity
  - Corruption perception
- Another subquery compares each group's average happiness against the overall average happiness score.

### Result Summary

In this dataset, the Low Corruption Perception group has a higher average happiness score than the High Corruption Perception group. This suggests that countries with lower corruption perception tend to show better happiness results.

