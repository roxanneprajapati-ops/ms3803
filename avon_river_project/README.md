# MSE803 Avon River Data Analytics Project

This project analyses Avon River water quality and fish population data for MSE803 Assessment 1. It uses Python for data cleaning, statistical analysis, machine learning, chart generation, CSV export, and dashboard data export.

## Project Structure

```text
avon_river_project/
│
├── main.py
├── data_cleaner.py
├── ml_analyzer.py
├── chart_generator.py
├── dashboard_exporter.py
├── requirements.txt
│
├── data/
│   └── Data_Set_Assignmnet_1-V0_1_20426.xlsx
│
├── output/
│   ├── water_quality_clean.csv
│   ├── fish_population_clean.csv
│   ├── avon_river_merged_clean.csv
│   ├── data.js
│   └── chart images
│
└── dashboard/
    └── dashboard.html
    └── style.css
    └── script.js
```

## What the Code Does

The pipeline follows these main steps:
1. Loads the raw Excel dataset.
2. Splits water quality and fish population data.
3. Cleans missing values, dates, duplicates, and inconsistent records.
4. Checks outliers using IQR and Z-score methods.
5. Merges the cleaned datasets.
6. Runs correlation analysis.
7. Applies K-Means clustering to group ecological health zones.
8. Uses Random Forest regression to predict fish count and calculate feature importance.
9. Runs dissolved oxygen trend analysis.
10. Generates charts for the report.
11. Exports cleaned CSV files.
12. Creates `data.js` for the HTML dashboard.

## Python Files

### `main.py`
Runs the full workflow in order.

### `data_cleaner.py`
Handles loading, splitting, cleaning, outlier checking, merging, and saving cleaned CSV files.

### `ml_analyzer.py`
Runs correlation analysis, K-Means clustering, Random Forest regression, and trend analysis.

### `chart_generator.py`
Creates the report charts. Table-style outputs are printed in the console instead of being saved as chart images.

### `dashboard_exporter.py`
Creates the `data.js` file used by the HTML dashboard.

## Installation

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Make sure the Excel dataset is inside the `data/` folder:

```text
data/dataset.xlsx
```

Then run:

```bash
python main.py
```

## Output Files

After running the code, the `output/` folder should contain:

```text
water_quality_clean.csv
fish_population_clean.csv
avon_river_merged_clean.csv
data.js
chart1_do_trend.png
chart2_fish_count.png
chart3_water_quality_dashboard.png
chart4_correlation_matrix.png
chart5_do_vs_fish_scatter.png
chart6_site_radar.png
chart7_parameter_change.png
chart9_do_projection.png
chart10_ml_insights.png
chart11_actual_vs_predicted.png
```

Risk summary and Project A vs Project B comparison are printed in the console, not saved as chart images.

## Machine Learning Used

Two machine learning methods are used:

### K-Means Clustering
Used to group river monitoring records into ecological health zones:

- Good
- Moderate
- At-Risk

### Random Forest Regression
Used to predict fish count based on:

- Temperature
- pH
- Dissolved Oxygen

It also shows feature importance, which helps identify which water quality factor has stronger influence on fish count.

## Notes
The dataset is small, so the ML results should be interpreted as supporting evidence, not as a final scientific prediction. The analysis is mainly used to identify patterns, risks, and recommendations for the conservation organisation.
