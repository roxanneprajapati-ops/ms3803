"""
World Happiness Dataset
Data cleaning and data visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load the CSV file."""
    return pd.read_csv(file_path)


def display_data(df, title):
    """Display first few rows of dataset."""
    print(f"\n{title}")
    print(df.head())


def check_missing_values(df):
    """Check missing values in each column."""
    print("\nMissing Values:")
    print(df.isnull().sum())


def fill_missing_values(df):
    """Fill missing values using mean for numbers and mode for text."""
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna(df[column].mode()[0])
        else:
            df[column] = df[column].fillna(df[column].mean())

    return df


def remove_outliers_iqr(df, numeric_columns):
    """Remove outliers using IQR method."""
    original_count = len(df)

    for column in numeric_columns:
        # Get lower and upper range
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        # Keep only normal values
        df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

    cleaned_count = len(df)

    print("\nOutlier Cleaning:")
    print("Records before cleaning:", original_count)
    print("Records after cleaning:", cleaned_count)
    print("Records removed:", original_count - cleaned_count)

    return df


def show_dataset_summary(df):
    """Show basic dataset information."""
    print("\nDataset Summary:")
    print(df.info())

    print("\nBasic Statistics:")
    print(df.describe())


def create_happiness_overview_image(df):
    """Create one image with multiple graphs about happiness factors."""

    # Sort data for bar chart
    df_sorted = df.sort_values(by="Happiness_Score", ascending=False)

    # Create one figure with 4 graphs
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Graph 1: Happiness score by country
    axes[0, 0].bar(df_sorted["Country"], df_sorted["Happiness_Score"])
    axes[0, 0].set_title("Happiness Score by Country")
    axes[0, 0].set_xlabel("Country")
    axes[0, 0].set_ylabel("Happiness Score")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Graph 2: GDP vs happiness
    axes[0, 1].scatter(df["GDP_per_Capita"], df["Happiness_Score"])
    axes[0, 1].set_title("GDP per Capita vs Happiness")
    axes[0, 1].set_xlabel("GDP per Capita")
    axes[0, 1].set_ylabel("Happiness Score")

    # Graph 3: Social support vs happiness
    axes[1, 0].scatter(df["Social_Support"], df["Happiness_Score"])
    axes[1, 0].set_title("Social Support vs Happiness")
    axes[1, 0].set_xlabel("Social Support")
    axes[1, 0].set_ylabel("Happiness Score")

    # Graph 4: Healthy life expectancy vs happiness
    axes[1, 1].scatter(df["Healthy_Life_Expectancy"], df["Happiness_Score"])
    axes[1, 1].set_title("Healthy Life Expectancy vs Happiness")
    axes[1, 1].set_xlabel("Healthy Life Expectancy")
    axes[1, 1].set_ylabel("Happiness Score")

    # Improve layout
    plt.tight_layout()

    # Save as one image
    plt.savefig("happiness_overview_graphs.png", dpi=300)

    # Show image
    plt.show()


def create_correlation_heatmap_image(df):
    """Create separate image for correlation heatmap."""

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Calculate correlation
    correlation = numeric_df.corr()

    plt.figure(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(correlation, annot=True, cmap="Blues")

    plt.title("Correlation Heatmap")

    # Improve layout
    plt.tight_layout()

    # Save as separate image
    plt.savefig("correlation_heatmap.png", dpi=300)

    # Show image
    plt.show()


def print_findings():
    """Print simple findings from the graphs."""
    print("\nFindings:")
    print("1. Some countries have higher happiness scores than others.")
    print("2. GDP per capita has a positive relationship with happiness.")
    print("3. Social support also affects happiness score.")
    print("4. Healthy life expectancy may help improve happiness.")
    print("5. Happiness is affected by many factors, not only income.")


def main():
    """Main function to run the full analysis."""

    # File name
    file_path = "world_happiness_dataset.csv"

    # Load dataset
    df = load_data(file_path)

    # Display original dataset
    display_data(df, "Original Dataset (Before Cleaning):")

    # Show missing values
    check_missing_values(df)

    # Fill missing values
    df = fill_missing_values(df)

    # Get numeric columns only
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns

    # Remove outliers
    df = remove_outliers_iqr(df, numeric_columns)

    # Display cleaned dataset
    display_data(df, "Cleaned Dataset (After Cleaning):")

    # Show clean data summary
    show_dataset_summary(df)

    # Create combined graph image
    create_happiness_overview_image(df)

    # Create heatmap image
    create_correlation_heatmap_image(df)

    # Print findings
    print_findings()


# Run the program
if __name__ == "__main__":
    main()