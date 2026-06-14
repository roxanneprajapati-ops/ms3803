"""
W10.A1.1
Linear Regression and Polynomial Regression
Salary Dataset Analysis
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_clean_data(file_path):
    """Load and clean the dataset."""

    df = pd.read_csv(file_path)

    # Remove unwanted index column if available
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove missing and duplicate rows
    df = df.dropna()
    df = df.drop_duplicates()

    return df


def evaluate_model(model_name, y_test, y_pred):
    """Calculate MAE, MSE, and RMSE."""

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{model_name} Results")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")

    return mae, mse, rmse


def predict_future_salaries(linear_model, poly, poly_model):
    """Predict salaries for 14, 14.5, and 15 years of experience."""

    experience_values = pd.DataFrame({
        "YearsExperience": [14, 14.5, 15]
    })

    # Predict using Linear Regression
    linear_predictions = linear_model.predict(experience_values)

    # Predict using Polynomial Regression
    experience_poly = poly.transform(experience_values)
    polynomial_predictions = poly_model.predict(experience_poly)

    prediction_df = pd.DataFrame({
        "YearsExperience": experience_values["YearsExperience"],
        "LinearRegressionSalary": np.round(linear_predictions, 2),
        "PolynomialRegressionSalary": np.round(polynomial_predictions, 2)
    })

    print("\nSalary Predictions")
    print(prediction_df)

    # Save prediction results
    prediction_df.to_csv("output/salary_predictions.csv", index=False)

    return prediction_df


def plot_dataset(df):
    """Save scatter plot of the original dataset."""

    plt.figure(figsize=(8, 5))

    # Plot actual salary data
    sns.scatterplot(
        data=df,
        x="YearsExperience",
        y="Salary"
    )

    plt.title("Years of Experience vs Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()

    plt.savefig("output/01_dataset_scatter.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_linear_regression(df, model):
    """Save linear regression chart."""

    plt.figure(figsize=(8, 5))

    # Plot actual data points
    sns.scatterplot(
        data=df,
        x="YearsExperience",
        y="Salary",
        label="Actual Data"
    )

    # Create DataFrame to avoid sklearn feature-name warning
    x_range = pd.DataFrame({
        "YearsExperience": np.linspace(
            df["YearsExperience"].min(),
            df["YearsExperience"].max(),
            100
        )
    })

    # Predict salary using Linear Regression
    y_line = model.predict(x_range)

    # Plot regression line
    sns.lineplot(
        x=x_range["YearsExperience"],
        y=y_line,
        label="Linear Regression"
    )

    plt.title("Linear Regression Model")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()

    plt.savefig("output/02_linear_regression.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_polynomial_regression(df, poly, model):
    """Save polynomial regression chart."""

    plt.figure(figsize=(8, 5))

    # Plot actual data points
    sns.scatterplot(
        data=df,
        x="YearsExperience",
        y="Salary",
        label="Actual Data"
    )

    # Create DataFrame to avoid sklearn feature-name warning
    x_range = pd.DataFrame({
        "YearsExperience": np.linspace(
            df["YearsExperience"].min(),
            df["YearsExperience"].max(),
            100
        )
    })

    # Transform feature into polynomial form
    x_poly = poly.transform(x_range)

    # Predict salary using Polynomial Regression
    y_curve = model.predict(x_poly)

    # Plot polynomial curve
    sns.lineplot(
        x=x_range["YearsExperience"],
        y=y_curve,
        label="Polynomial Regression"
    )

    plt.title("Polynomial Regression Model Degree 2")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()

    plt.savefig("output/03_polynomial_regression.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_model_comparison(results_df):
    """Save model error comparison chart."""

    results_melted = results_df.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Error Value"
    )

    plt.figure(figsize=(8, 5))

    # Compare MAE and RMSE values
    sns.barplot(
        data=results_melted,
        x="Metric",
        y="Error Value",
        hue="Model"
    )

    plt.title("Model Error Comparison")
    plt.xlabel("Error Metric")
    plt.ylabel("Error Value")
    plt.tight_layout()

    plt.savefig("output/04_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_salary_predictions(df, prediction_df, linear_model, poly, poly_model):
    """Save chart showing historical data and future salary predictions."""

    plt.figure(figsize=(10, 6))

    # Plot historical salary data from the dataset
    sns.scatterplot(
        data=df,
        x="YearsExperience",
        y="Salary",
        label="Historical Data",
        s=70
    )

    # Create smooth experience range from dataset minimum up to 15 years
    x_range = pd.DataFrame({
        "YearsExperience": np.linspace(
            df["YearsExperience"].min(),
            15,
            200
        )
    })

    # Predict salary using Linear Regression
    linear_line = linear_model.predict(x_range)

    # Predict salary using Polynomial Regression
    x_range_poly = poly.transform(x_range)
    polynomial_curve = poly_model.predict(x_range_poly)

    # Plot Linear Regression prediction line
    sns.lineplot(
        x=x_range["YearsExperience"],
        y=linear_line,
        label="Linear Regression Line"
    )

    # Plot Polynomial Regression prediction curve
    sns.lineplot(
        x=x_range["YearsExperience"],
        y=polynomial_curve,
        label="Polynomial Regression Curve"
    )

    # Plot future prediction points from Linear Regression
    sns.scatterplot(
        data=prediction_df,
        x="YearsExperience",
        y="LinearRegressionSalary",
        label="Linear Predictions",
        marker="X",
        s=120
    )

    # Plot future prediction points from Polynomial Regression
    sns.scatterplot(
        data=prediction_df,
        x="YearsExperience",
        y="PolynomialRegressionSalary",
        label="Polynomial Predictions",
        marker="D",
        s=100
    )

    # Add salary labels for Linear Regression predictions
    for _, row in prediction_df.iterrows():
        plt.text(
            row["YearsExperience"],
            row["LinearRegressionSalary"],
            f'{row["LinearRegressionSalary"]:,.0f}',
            ha="center",
            va="bottom",
            fontsize=8
        )

    # Add salary labels for Polynomial Regression predictions
    for _, row in prediction_df.iterrows():
        plt.text(
            row["YearsExperience"],
            row["PolynomialRegressionSalary"],
            f'{row["PolynomialRegressionSalary"]:,.0f}',
            ha="center",
            va="top",
            fontsize=8
        )

    plt.title("Historical Salary Data with Future Salary Predictions")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()

    plt.savefig(
        "output/05_salary_predictions_with_history.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


def main():
    """Run the complete regression analysis."""

    # Create output folder for charts and CSV files
    os.makedirs("output", exist_ok=True)

    # Load and clean dataset
    df = load_and_clean_data("salary-dataset.csv")

    print("Cleaned Dataset:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    # Select independent variable and target variable
    X = df[["YearsExperience"]]
    y = df["Salary"]

    # Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # =========================
    # Linear Regression
    # =========================

    linear_model = LinearRegression()

    # Train Linear Regression model
    linear_model.fit(X_train, y_train)

    # Predict test values
    y_pred_linear = linear_model.predict(X_test)

    # Evaluate Linear Regression model
    linear_metrics = evaluate_model(
        "Linear Regression",
        y_test,
        y_pred_linear
    )

    # =========================
    # Polynomial Regression
    # =========================

    poly = PolynomialFeatures(degree=2)

    # Convert feature into polynomial features
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()

    # Train Polynomial Regression model
    poly_model.fit(X_train_poly, y_train)

    # Predict test values
    y_pred_poly = poly_model.predict(X_test_poly)

    # Evaluate Polynomial Regression model
    poly_metrics = evaluate_model(
        "Polynomial Regression",
        y_test,
        y_pred_poly
    )

    # Store model comparison results
    results_df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Polynomial Regression"
        ],
        "MAE": [
            linear_metrics[0],
            poly_metrics[0]
        ],
        "MSE": [
            linear_metrics[1],
            poly_metrics[1]
        ],
        "RMSE": [
            linear_metrics[2],
            poly_metrics[2]
        ]
    })

    print("\nModel Comparison:")
    print(results_df)

    # Save model comparison results
    results_df.to_csv("output/model_comparison_results.csv", index=False)

    # Print conclusion based on RMSE
    if poly_metrics[2] < linear_metrics[2]:
        print("\nPolynomial Regression performed better based on RMSE.")
    else:
        print("\nLinear Regression performed better based on RMSE.")

    # Predict salaries for 14, 14.5, and 15 years of experience
    prediction_df = predict_future_salaries(
        linear_model,
        poly,
        poly_model
    )

    # Generate and save charts
    plot_dataset(df)
    plot_linear_regression(df, linear_model)
    plot_polynomial_regression(df, poly, poly_model)
    plot_model_comparison(results_df)
    plot_salary_predictions(
        df,
        prediction_df,
        linear_model,
        poly,
        poly_model
    )

    print("\nAll charts and CSV results are saved in the output folder.")


if __name__ == "__main__":
    main()