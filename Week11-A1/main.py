import pandas as pd
import numpy as np


def load_dataset(url):
    """Load dataset from URL."""
    df = pd.read_csv(url)
    return df


def preprocess_data(df):
    """Clean and prepare the dataset."""

    # Convert Month column to datetime
    df["Month"] = pd.to_datetime(df["Month"])

    # Set Month as index
    df.set_index("Month", inplace=True)

    # Rename column
    df.rename(columns={"Passengers": "passengers"}, inplace=True)

    # Add time-based features
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["time_index"] = np.arange(len(df))

    # Add lag features
    df["lag_1"] = df["passengers"].shift(1)
    df["lag_2"] = df["passengers"].shift(2)
    df["lag_3"] = df["passengers"].shift(3)
    df["lag_12"] = df["passengers"].shift(12)

    # Add rolling mean features
    df["rolling_mean_3"] = df["passengers"].rolling(window=3).mean()
    df["rolling_mean_12"] = df["passengers"].rolling(window=12).mean()

    # Drop missing values created by lag and rolling features
    df_preprocessed = df.dropna()

    return df_preprocessed


def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"

    # Load data
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

    # Preprocess data
    df_preprocessed = preprocess_data(df)

    print("\nPreprocessed dataset:")
    print(df_preprocessed.head())

    print("\nPreprocessed dataset shape:")
    print(df_preprocessed.shape)


if __name__ == "__main__":
    main()