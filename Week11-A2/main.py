"""
Week 11 - Activity 2
New Zealand Wellbeing Statistics 2014-18 Forecasting Project

This single Python file:
1. Loads the Stats NZ wellbeing Excel dataset
2. Extracts Total Population data
3. Prepares and processes time-series data
4. Applies Linear Regression, XGBoost, ANN, LSTM, and ARIMA
5. Evaluates each model using MAE, RMSE, and MAPE
6. Identifies the best model using lowest RMSE
7. Creates simplified useful charts for analysis

Author: Roxanne Peñaverde Prajapati
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False


DATA_FILE = "wellbeing-statistics-2014-18-time-series.xlsx"
SHEET_NAME = "1. Overall life satisfaction"

OUTPUT_FOLDER = "output"
CHART_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
RESULT_FOLDER = os.path.join(OUTPUT_FOLDER, "results")

os.makedirs(CHART_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(
        np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 0.00001))
    ) * 100


def evaluate_model(model_name, y_true, y_pred):
    """Evaluate each model using MAE, RMSE, and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)

    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4)
    }


def load_total_population_data(file_path, sheet_name):
    """
    Load the Excel file and extract Total Population wellbeing values.
    """
    print("STEP 1: LOAD DATA")

    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    print(f"Loaded sheet: {sheet_name}")
    print(f"Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")

    total_population_row = None

    for index, row in raw.iterrows():
        row_text = " ".join(row.astype(str).tolist())

        if "Total population" in row_text:
            total_population_row = index
            break

    if total_population_row is None:
        raise ValueError("Could not find 'Total population' in the selected sheet.")

    print(f"Total population section found at row: {total_population_row}")

    data_rows = raw.iloc[total_population_row + 1: total_population_row + 4].copy()

    years = pd.to_numeric(data_rows.iloc[:, 1], errors="coerce")

    possible_target_columns = []

    for col in range(raw.shape[1]):
        values = pd.to_numeric(data_rows.iloc[:, col], errors="coerce")

        if values.notna().sum() >= 3:
            possible_target_columns.append(col)

    if 17 in possible_target_columns:
        target_column = 17
    else:
        target_column = possible_target_columns[-1]

    values = pd.to_numeric(data_rows.iloc[:, target_column], errors="coerce")

    df = pd.DataFrame({
        "Year": years,
        "Value": values
    })

    df = df.dropna()
    df["Year"] = df["Year"].astype(int)
    df["Value"] = df["Value"].astype(float)
    df = df.sort_values("Year")

    print("\nExtracted data:")
    print(df)

    return df



def prepare_data(df):
    """
    Prepare data for forecasting.

    Processing steps:
    1. Sort by year
    2. Add missing years
    3. Interpolate missing values
    4. Create Time_Index
    5. Create Lag_1
    """

    print("STEP 2: DATA PREPARATION AND PROCESSING")

    df = df.copy()
    df = df.sort_values("Year")

    full_years = pd.DataFrame({
        "Year": np.arange(df["Year"].min(), df["Year"].max() + 1)
    })

    processed = full_years.merge(df, on="Year", how="left")

    processed["Data_Type"] = np.where(
        processed["Value"].isna(),
        "Interpolated",
        "Original"
    )

    processed["Value"] = processed["Value"].interpolate(method="linear")

    processed["Time_Index"] = processed["Year"] - processed["Year"].min()

    processed["Lag_1"] = processed["Value"].shift(1)
    processed["Lag_1"] = processed["Lag_1"].bfill()

    print("\nProcessed data:")
    print(processed)

    processed.to_csv(
        os.path.join(RESULT_FOLDER, "processed_wellbeing_data.csv"),
        index=False
    )

    return processed

def create_train_test_data(df):
    """
    Split data into training and testing.

    Train: 2014 to 2017
    Test: 2018
    """

    print("\n========================================")
    print("STEP 3: TRAIN-TEST SPLIT")
    print("========================================")

    features = ["Time_Index", "Lag_1"]

    train = df[df["Year"] < df["Year"].max()].copy()
    test = df[df["Year"] == df["Year"].max()].copy()

    X_train = train[features]
    y_train = train["Value"]

    X_test = test[features]
    y_test = test["Value"]

    print(f"Training rows: {len(train)}")
    print(f"Testing rows: {len(test)}")
    print(f"Features used: {features}")

    return train, test, X_train, X_test, y_train, y_test


def run_linear_regression(X_train, X_test, y_train):
    """Run Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    return prediction, model


def run_xgboost(X_train, X_test, y_train):
    """Run XGBoost model."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Skipping XGBoost.")
        return None, None

    model = XGBRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    return prediction, model


def run_ann(X_train, X_test, y_train):
    """Run Artificial Neural Network model."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not installed. Skipping ANN.")
        return None, None

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(
        np.array(y_train).reshape(-1, 1)
    )

    model = Sequential()
    model.add(Dense(12, activation="relu", input_shape=(X_train_scaled.shape[1],)))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mse"
    )

    model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=150,
        verbose=0
    )

    prediction_scaled = model.predict(X_test_scaled, verbose=0)
    prediction = scaler_y.inverse_transform(prediction_scaled).flatten()

    return prediction, model


def run_lstm(train_values):
    """
    Run LSTM model.

    LSTM is included because the activity requires it.
    The dataset is small, so result reliability is limited.
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not installed. Skipping LSTM.")
        return None, None

    scaler = MinMaxScaler()

    values = np.array(train_values).reshape(-1, 1)
    scaled_values = scaler.fit_transform(values)

    X_seq = []
    y_seq = []

    for i in range(1, len(scaled_values)):
        X_seq.append(scaled_values[i - 1:i, 0])
        y_seq.append(scaled_values[i, 0])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

    model = Sequential()
    model.add(LSTM(12, activation="relu", input_shape=(X_seq.shape[1], 1)))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mse"
    )

    model.fit(
        X_seq,
        y_seq,
        epochs=150,
        verbose=0
    )

    last_value = scaled_values[-1].reshape(1, 1, 1)

    prediction_scaled = model.predict(last_value, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled).flatten()

    return prediction, model


def run_arima(train_values):
    """Run ARIMA model."""
    if not ARIMA_AVAILABLE:
        print("Statsmodels is not installed. Skipping ARIMA.")
        return None, None

    try:
        model = ARIMA(train_values, order=(1, 1, 0))
        fitted_model = model.fit()

        prediction = np.array(fitted_model.forecast(steps=1))

        return prediction, fitted_model

    except Exception as error:
        print(f"ARIMA failed: {error}")
        return None, None

def forecast_next_year(best_model_name, models, processed_df):
    """Forecast the year after the final available year."""

    last_row = processed_df.iloc[-1]

    next_year = int(last_row["Year"] + 1)
    next_time_index = int(last_row["Time_Index"] + 1)
    next_lag_1 = float(last_row["Value"])

    next_data = pd.DataFrame({
        "Time_Index": [next_time_index],
        "Lag_1": [next_lag_1]
    })

    if best_model_name == "Linear Regression":
        forecast_value = models["Linear Regression"].predict(next_data)[0]

    elif best_model_name == "XGBoost" and models["XGBoost"] is not None:
        forecast_value = models["XGBoost"].predict(next_data)[0]

    elif best_model_name == "ARIMA" and models["ARIMA"] is not None:
        try:
            forecast_value = models["ARIMA"].forecast(steps=2)[-1]
        except Exception:
            forecast_value = models["Linear Regression"].predict(next_data)[0]

    else:
        forecast_value = models["Linear Regression"].predict(next_data)[0]

    forecast_df = pd.DataFrame({
        "Year": [next_year],
        "Forecasted_Value": [round(float(forecast_value), 4)],
        "Model_Used": [best_model_name]
    })

    forecast_df.to_csv(
        os.path.join(RESULT_FOLDER, "next_year_forecast.csv"),
        index=False
    )

    return forecast_df

def save_historical_trend_chart(df):
    """Save historical trend chart."""

    plt.figure(figsize=(9, 5))

    plt.plot(df["Year"], df["Value"], marker="o", linewidth=2)

    for _, row in df.iterrows():
        plt.text(
            row["Year"],
            row["Value"],
            f'{row["Value"]:.2f}',
            ha="center",
            va="bottom"
        )

    plt.title("Historical Wellbeing Trend")
    plt.xlabel("Year")
    plt.ylabel("Wellbeing Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_FOLDER, "01_historical_trend.png"),
        dpi=300
    )
    plt.close()


def save_correlation_heatmap(df):
    """Save correlation heatmap."""

    numeric_df = df[["Year", "Value", "Time_Index", "Lag_1"]]
    corr = numeric_df.corr()

    plt.figure(figsize=(7, 5))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")

    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.title("Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_FOLDER, "02_correlation_heatmap.png"),
        dpi=300
    )
    plt.close()


def save_actual_vs_predicted_chart(test_year, y_true, predictions):
    """
    Save improved Actual vs Predicted chart.

    The y-axis is zoomed in because wellbeing values are close together.
    """

    model_names = ["Actual"]
    values = [float(y_true.iloc[0])]

    for model_name, prediction in predictions.items():
        if prediction is not None:
            model_names.append(model_name)
            values.append(float(prediction[0]))

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, values)

    plt.title(f"Actual vs Predicted Wellbeing Value ({test_year})")
    plt.xlabel("Model")
    plt.ylabel("Wellbeing Value")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    # Zoom y-axis so small differences are visible
    plt.ylim(min(values) - 0.05, max(values) + 0.05)

    for i, value in enumerate(values):
        plt.text(
            i,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_FOLDER, "03_actual_vs_predicted_zoomed.png"),
        dpi=300
    )
    plt.close()


def save_absolute_error_chart(prediction_df):
    """
    Save absolute error chart.

    This is easier to interpret than Actual vs Predicted
    when all values are very close.
    """

    error_df = prediction_df.copy()
    error_df["Absolute_Error"] = abs(
        error_df["Actual_Value"] - error_df["Predicted_Value"]
    )

    plt.figure(figsize=(9, 5))
    plt.bar(error_df["Model"], error_df["Absolute_Error"])

    plt.title("Model Absolute Error Comparison")
    plt.xlabel("Model")
    plt.ylabel("Absolute Error")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(error_df["Absolute_Error"]):
        plt.text(
            i,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_FOLDER, "04_absolute_error_comparison.png"),
        dpi=300
    )
    plt.close()

    error_df.to_csv(
        os.path.join(RESULT_FOLDER, "absolute_error_comparison.csv"),
        index=False
    )


def save_model_performance_chart(results_df):
    """Save grouped model performance chart."""

    metrics = ["MAE", "RMSE", "MAPE"]
    x = np.arange(len(results_df["Model"]))
    width = 0.25

    plt.figure(figsize=(11, 5))

    for i, metric in enumerate(metrics):
        plt.bar(
            x + (i - 1) * width,
            results_df[metric],
            width,
            label=metric
        )

    plt.title("Model Performance Comparison")
    plt.xlabel("Model")
    plt.ylabel("Error Value")
    plt.xticks(x, results_df["Model"], rotation=30, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_FOLDER, "05_model_performance.png"),
        dpi=300
    )
    plt.close()


def save_next_year_forecast_chart(df, forecast_df):
    """Save historical and next-year forecast chart."""

    forecast_year = forecast_df["Year"].iloc[0]
    forecast_value = forecast_df["Forecasted_Value"].iloc[0]

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["Year"],
        df["Value"],
        marker="o",
        linewidth=2,
        label="Historical"
    )

    plt.scatter(
        [forecast_year],
        [forecast_value],
        s=100,
        label="Forecast"
    )

    plt.plot(
        [df["Year"].iloc[-1], forecast_year],
        [df["Value"].iloc[-1], forecast_value],
        linestyle="--"
    )

    plt.text(
        forecast_year,
        forecast_value,
        f"{forecast_value:.2f}",
        ha="center",
        va="bottom"
    )

    plt.title("Next-Year Wellbeing Forecast")
    plt.xlabel("Year")
    plt.ylabel("Wellbeing Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(CHART_FOLDER, "06_next_year_forecast.png"),
        dpi=300
    )
    plt.close()


def main():
    """Run the full forecasting activity."""

    if not os.path.exists(DATA_FILE):
        print("\nERROR: Dataset file not found.")
        print("Place this Excel file in the same folder as this Python script:")
        print(DATA_FILE)
        return

    original_df = load_total_population_data(DATA_FILE, SHEET_NAME)
    processed_df = prepare_data(original_df)

    train, test, X_train, X_test, y_train, y_test = create_train_test_data(
        processed_df
    )

    print("STEP 4: MODEL TRAINING")
    predictions = {}
    models = {}

    lr_pred, lr_model = run_linear_regression(X_train, X_test, y_train)
    predictions["Linear Regression"] = lr_pred
    models["Linear Regression"] = lr_model
    print("Linear Regression completed.")

    xgb_pred, xgb_model = run_xgboost(X_train, X_test, y_train)
    predictions["XGBoost"] = xgb_pred
    models["XGBoost"] = xgb_model

    if xgb_pred is not None:
        print("XGBoost completed.")

    ann_pred, ann_model = run_ann(X_train, X_test, y_train)
    predictions["ANN"] = ann_pred
    models["ANN"] = ann_model

    if ann_pred is not None:
        print("ANN completed.")

    lstm_pred, lstm_model = run_lstm(train["Value"].values)
    predictions["LSTM"] = lstm_pred
    models["LSTM"] = lstm_model

    if lstm_pred is not None:
        print("LSTM completed.")

    arima_pred, arima_model = run_arima(train["Value"].values)
    predictions["ARIMA"] = arima_pred
    models["ARIMA"] = arima_model

    if arima_pred is not None:
        print("ARIMA completed.")

    print("STEP 5: MODEL EVALUATION")
    results = []

    for model_name, prediction in predictions.items():
        if prediction is not None:
            results.append(
                evaluate_model(model_name, y_test, prediction)
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")

    results_df.to_csv(
        os.path.join(RESULT_FOLDER, "model_performance.csv"),
        index=False
    )

    print("\nModel Performance Comparison:")
    print(results_df)

    best_model = results_df.iloc[0]["Model"]

    print("\nBest-performing model based on lowest RMSE:")
    print(best_model)

    prediction_rows = []

    for model_name, prediction in predictions.items():
        if prediction is not None:
            prediction_rows.append({
                "Year": int(test["Year"].iloc[0]),
                "Actual_Value": round(float(y_test.iloc[0]), 4),
                "Model": model_name,
                "Predicted_Value": round(float(prediction[0]), 4)
            })

    prediction_df = pd.DataFrame(prediction_rows)

    prediction_df.to_csv(
        os.path.join(RESULT_FOLDER, "predictions.csv"),
        index=False
    )

    print("\nPredictions:")
    print(prediction_df)

    forecast_df = forecast_next_year(best_model, models, processed_df)

    print("\nNext-Year Forecast:")
    print(forecast_df)

    print("STEP 6: GENERATING CHARTS")
    save_historical_trend_chart(processed_df)
    save_correlation_heatmap(processed_df)
    save_actual_vs_predicted_chart(
        int(test["Year"].iloc[0]),
        y_test,
        predictions
    )
    save_absolute_error_chart(prediction_df)
    save_model_performance_chart(results_df)
    save_next_year_forecast_chart(processed_df, forecast_df)

    print("\nCharts saved in:")
    print(CHART_FOLDER)

    print("\nResults saved in:")
    print(RESULT_FOLDER)


    print("TOP 3 FINDINGS")
    print("1. The selected wellbeing measure stayed mostly stable from 2014 to 2018.")
    print("2. Simple models are more suitable because the dataset has very few time points.")
    print(f"3. The best-performing model based on RMSE is {best_model}.")

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()