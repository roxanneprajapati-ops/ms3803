# pip install pandas numpy matplotlib scikit-learn xgboost statsmodels
# Optional for LSTM only: pip install tensorflow

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

# TensorFlow is optional because sometimes it is not installed
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam
except ModuleNotFoundError:
    Sequential = None
    Dense = None
    LSTM = None
    Adam = None


def load_dataset(url):
    """Load dataset from URL."""

    # Read CSV file from online link
    df = pd.read_csv(url)

    # Return raw dataset
    return df


def preprocess_data(df):
    """Clean and prepare dataset."""

    # Convert Month column to date format
    df["Month"] = pd.to_datetime(df["Month"])

    # Set Month as index because this is time series data
    df.set_index("Month", inplace=True)

    # Rename column to simple name
    df.rename(columns={"Passengers": "passengers"}, inplace=True)

    # Create year feature
    df["year"] = df.index.year

    # Create month feature
    df["month"] = df.index.month

    # Create time order number
    df["time_index"] = np.arange(len(df))

    # Create previous month passenger values
    df["lag_1"] = df["passengers"].shift(1)
    df["lag_2"] = df["passengers"].shift(2)
    df["lag_3"] = df["passengers"].shift(3)

    # Create same month last year value
    df["lag_12"] = df["passengers"].shift(12)

    # Create rolling average values
    df["rolling_mean_3"] = df["passengers"].rolling(3).mean()
    df["rolling_mean_12"] = df["passengers"].rolling(12).mean()

    # Remove rows with empty values from lag and rolling
    df = df.dropna()

    return df


def perform_eda(df):
    """Create and save EDA charts."""

    # Create chart folder
    os.makedirs("output/charts", exist_ok=True)

    # Figure 1: Monthly passenger trend
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["passengers"], linewidth=2)
    plt.title("Monthly Airline Passenger Trend")
    plt.xlabel("Year")
    plt.ylabel("Passengers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "output/charts/figure_1_monthly_passenger_trend.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    # Figure 2: Total passengers per year
    yearly = df.groupby(df.index.year)["passengers"].sum()

    plt.figure(figsize=(10, 5))
    yearly.plot(kind="bar")
    plt.title("Total Airline Passengers Per Year")
    plt.xlabel("Year")
    plt.ylabel("Total Passengers")
    plt.tight_layout()
    plt.savefig(
        "output/charts/figure_2_total_passengers_per_year.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


def split_ml_data(df):
    """Split data for ML models."""

    # Input features for ML models
    features = [
        "year",
        "month",
        "time_index",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_12"
    ]

    # X is input
    X = df[features]

    # y is target to predict
    y = df["passengers"]

    # Use first 80% for training, last 20% for testing
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def evaluate_model(model_name, y_test, y_pred):
    """Calculate model evaluation result."""

    # Calculate errors
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Return result as dictionary
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2)
    }


def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression model."""

    # Create model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluate result
    result = evaluate_model("Linear Regression", y_test, y_pred)

    return y_pred, result


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model."""

    # Create XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Evaluate result
    result = evaluate_model("XGBoost", y_test, y_pred)

    return y_pred, result


def train_ann(X_train, X_test, y_train, y_test):
    """Train ANN using MLPRegressor."""

    # Scale input data because ANN works better with scaled data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create ANN model
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=2000,
        random_state=42
    )

    # Train ANN model
    model.fit(X_train_scaled, y_train)

    # Predict test data
    y_pred = model.predict(X_test_scaled)

    # Evaluate result
    result = evaluate_model("ANN", y_test, y_pred)

    return y_pred, result


def create_lstm_sequences(data, window_size=12):
    """Create sequence data for LSTM."""

    X = []
    y = []

    # Create 12 months input to predict next month
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def train_lstm(df, y_test):
    """Train LSTM model."""

    # Skip LSTM if TensorFlow is not installed
    if Sequential is None:
        print("\nTensorFlow is not installed. LSTM model skipped.")
        print("To run LSTM, install TensorFlow using: pip install tensorflow")
        return None, None

    # Use passenger column only for LSTM
    values = df["passengers"].values.reshape(-1, 1)

    # Scale passenger values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Create LSTM sequence
    window_size = 12
    X_lstm, y_lstm = create_lstm_sequences(scaled_values, window_size)

    # Reshape data for LSTM
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

    # Match test size with other models
    test_size = len(y_test)

    X_train = X_lstm[:-test_size]
    X_test = X_lstm[-test_size:]
    y_train = y_lstm[:-test_size]

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(window_size, 1)))
    model.add(Dense(1))

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss="mse"
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=8,
        verbose=0
    )

    # Predict test data
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Convert prediction back to original passenger scale
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    # Evaluate result
    result = evaluate_model("LSTM", y_test, y_pred)

    return y_pred, result


def train_arima(y_train, y_test):
    """Train ARIMA model."""

    # Create ARIMA model
    model = ARIMA(y_train, order=(2, 1, 2))

    # Train model
    fitted_model = model.fit()

    # Forecast same length as test data
    y_pred = fitted_model.forecast(steps=len(y_test))

    # Evaluate result
    result = evaluate_model("ARIMA", y_test, y_pred)

    return y_pred, result


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Create actual vs predicted chart."""

    # Skip if prediction is empty
    if y_pred is None:
        return

    # Create chart folder
    os.makedirs("output/charts", exist_ok=True)

    # Create chart
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label="Actual", linewidth=2)
    plt.plot(y_test.index, y_pred, label="Predicted", linewidth=2)

    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Year")
    plt.ylabel("Passengers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save chart
    file_name = model_name.lower().replace(" ", "_")
    plt.savefig(
        f"output/charts/actual_vs_predicted_{file_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def save_results(results_df):
    """Save model comparison result."""

    # Create report folder
    os.makedirs("output/reports", exist_ok=True)

    # Save result table
    results_df.to_csv(
        "output/reports/model_performance_results.csv",
        index=False
    )

    print("\nModel results saved to:")
    print("output/reports/model_performance_results.csv")


def generate_findings_support(df, results_df):
    """Generate extra analysis to support findings."""

    # Create output folders
    os.makedirs("output/reports", exist_ok=True)
    os.makedirs("output/charts", exist_ok=True)

    # -----------------------------
    # Finding 1: Yearly growth
    # -----------------------------

    # Get yearly total passengers
    yearly_passengers = df.groupby(df.index.year)["passengers"].sum()

    # Calculate yearly growth percentage
    yearly_growth = yearly_passengers.pct_change() * 100

    # Create summary table
    growth_summary = pd.DataFrame({
        "Year": yearly_passengers.index,
        "Total_Passengers": yearly_passengers.values,
        "Growth_Percentage": yearly_growth.round(2).values
    })

    # Save yearly growth summary
    growth_summary.to_csv(
        "output/reports/yearly_growth_summary.csv",
        index=False
    )

    print("\nFinding Support 1: Yearly Growth Summary")
    print(growth_summary)

    # -----------------------------
    # Finding 2: Monthly seasonality
    # -----------------------------

    # Get average passengers per month
    monthly_average = df.groupby(df.index.month)["passengers"].mean()

    # Save monthly summary
    monthly_average.to_csv(
        "output/reports/monthly_seasonality_summary.csv"
    )

    # Create monthly seasonality chart
    plt.figure(figsize=(10, 5))
    monthly_average.plot(kind="bar")
    plt.title("Average Passenger Count by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Passengers")
    plt.tight_layout()
    plt.savefig(
        "output/charts/figure_5_average_passengers_by_month.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    print("\nFinding Support 2: Monthly Seasonality Summary")
    print(monthly_average)

    # -----------------------------
    # Finding 3: Lag feature relationship
    # -----------------------------

    # Check correlation of passenger with lag and rolling features
    lag_correlation = df[
        [
            "passengers",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_12",
            "rolling_mean_3",
            "rolling_mean_12"
        ]
    ].corr()["passengers"].sort_values(ascending=False)

    # Save correlation result
    lag_correlation.to_csv(
        "output/reports/lag_feature_correlation.csv"
    )

    print("\nFinding Support 3: Lag Feature Correlation")
    print(lag_correlation)

    # -----------------------------
    # Best model result
    # -----------------------------

    # Sort by RMSE because lower RMSE is better
    best_model = results_df.sort_values("RMSE").iloc[0]

    # Save best model summary
    best_model.to_csv(
        "output/reports/best_model_summary.csv"
    )

    print("\nBest Model Based on RMSE:")
    print(best_model)


def main():
    """Main function to run all steps."""

    # Dataset online link
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

    # -----------------------------
    # Step 1: Load data
    # -----------------------------

    df = load_dataset(url)

    print("First 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nBasic statistics:")
    print(df.describe())

    # -----------------------------
    # Step 2: Preprocess data
    # -----------------------------

    df = preprocess_data(df)

    print("\nPreprocessed data:")
    print(df.head())

    print("\nPreprocessed shape:")
    print(df.shape)

    # -----------------------------
    # Step 3: EDA graphs
    # -----------------------------

    perform_eda(df)

    # -----------------------------
    # Step 4: Split data
    # -----------------------------

    X_train, X_test, y_train, y_test = split_ml_data(df)

    print("\nTraining set size:")
    print(X_train.shape)

    print("\nTesting set size:")
    print(X_test.shape)

    # Store all model results
    results = []

    # -----------------------------
    # Step 5: Linear Regression
    # -----------------------------

    lr_pred, lr_result = train_linear_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(lr_result)
    plot_actual_vs_predicted(y_test, lr_pred, "Linear Regression")

    # -----------------------------
    # Step 6: XGBoost
    # -----------------------------

    xgb_pred, xgb_result = train_xgboost(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(xgb_result)
    plot_actual_vs_predicted(y_test, xgb_pred, "XGBoost")

    # -----------------------------
    # Step 7: ANN
    # -----------------------------

    ann_pred, ann_result = train_ann(
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(ann_result)
    plot_actual_vs_predicted(y_test, ann_pred, "ANN")

    # -----------------------------
    # Step 8: LSTM
    # -----------------------------

    lstm_pred, lstm_result = train_lstm(df, y_test)

    # Add LSTM only if TensorFlow is installed
    if lstm_result is not None:
        results.append(lstm_result)
        plot_actual_vs_predicted(y_test, lstm_pred, "LSTM")

    # -----------------------------
    # Step 9: ARIMA
    # -----------------------------

    arima_pred, arima_result = train_arima(y_train, y_test)

    results.append(arima_result)
    plot_actual_vs_predicted(y_test, arima_pred, "ARIMA")

    # -----------------------------
    # Step 10: Compare models
    # -----------------------------

    results_df = pd.DataFrame(results)

    print("\nModel Performance Comparison:")
    print(results_df)

    save_results(results_df)

    # -----------------------------
    # Step 11: Support findings
    # -----------------------------

    generate_findings_support(df, results_df)

    print("\nAll charts saved in: output/charts")
    print("All reports saved in: output/reports")


if __name__ == "__main__":
    main()