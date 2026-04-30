import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load the raw CSV dataset."""
    df = pd.read_csv(file_path)

    print("\nSTEP 1 — LOAD DATA")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nRaw Data:")
    print(df)

    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    return df


def data_scrubbing(df):
    """
    Data Scrubbing:
    Fix dirty data like text numbers, invalid dates,
    duplicate IDs, and messy salary values.
    """
    df = df.copy()

    # This is key-value pair dictionary.
    # Key is word number, value is real number.
    # Example: "thirty" = 30, "eight" = 8
    # So "thirty-eight" becomes 30 + 8 = 38.
    word_to_int = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }

    def words_to_number(value):
        """Convert text age like 'thirty-eight' to number 38."""
        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower()

        # If already number, just convert to float.
        try:
            return float(value)
        except ValueError:
            total = 0

            # Split words.
            # Example: "thirty-eight" becomes "thirty eight"
            for word in value.replace("-", " ").split():
                total += word_to_int.get(word, 0)

            return float(total) if total > 0 else np.nan

    def clean_salary(value):
        """Convert messy salary into numeric salary."""
        if pd.isna(value):
            return np.nan

        # Remove comma and dollar sign if there is any.
        value = str(value).strip().lower().replace(",", "").replace("$", "")

        # If salary is already number, convert directly.
        try:
            return float(value)
        except ValueError:
            total = 0
            multiplier = 1

            # Example: "sixty five thousand"
            # sixty + five = 65, then thousand means x 1000.
            for word in value.split():
                if word == "thousand":
                    multiplier = 1000
                else:
                    total += word_to_int.get(word, 0)

            return float(total * multiplier) if total > 0 else np.nan

    # Clean Age and Salary columns.
    df["Age"] = df["Age"].apply(words_to_number)
    df["Salary"] = df["Salary"].apply(clean_salary)

    # Convert ID to number so duplicate ID can be checked.
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")

    # Find duplicate IDs.
    duplicated_ids = df[df.duplicated("ID", keep=False) & df["ID"].notna()]["ID"].unique()

    merged_rows = []

    # Merge duplicate rows by keeping first available value.
    for duplicate_id in duplicated_ids:
        group = df[df["ID"] == duplicate_id]
        merged = {}

        for col in group.columns:
            non_null_values = group[col].dropna()
            merged[col] = non_null_values.iloc[0] if len(non_null_values) > 0 else np.nan

        merged_rows.append(merged)

    # Remove old duplicate rows.
    df = df.drop(df[df["ID"].isin(duplicated_ids)].index)

    # Add merged rows back.
    if merged_rows:
        df = pd.concat([df, pd.DataFrame(merged_rows)], ignore_index=True)

    # Convert Join Date to real date.
    # If date is invalid, it becomes NaT.
    df["Join Date"] = pd.to_datetime(df["Join Date"], errors="coerce", dayfirst=True)

    print("\nSTEP 2A — DATA SCRUBBING COMPLETE")
    return df


def data_munging(df):
    """
    Data Munging:
    Make data format consistent.
    """
    df = df.copy()

    # Country mapping.
    # This makes country values consistent.
    country_map = {
        "AU": "AUS",
        "AUS": "AUS",
        "NZ": "NZL",
        "NZL": "NZL"
    }

    df["Country"] = df["Country"].astype(str).str.strip().str.upper().map(country_map)

    # Clean name format.
    df["Name"] = df["Name"].astype(str).str.strip().str.title()
    df["Name"] = df["Name"].replace("Nan", np.nan)

    # Make sure these columns are correct data type.
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

    print("\nSTEP 2B — DATA MUNGING COMPLETE")
    return df


def data_wrangling(df):
    """
    Data Wrangling:
    Fill missing values, create Tenure_Years,
    and add outlier flags.
    """
    df = df.copy()

    # Use median because it is safer if there are extreme values.
    age_median = df["Age"].median()
    salary_median = df["Salary"].median()

    # Use mode because Country is text/category.
    country_mode = df["Country"].mode()[0]

    # Use middle date to fill missing Join Date.
    median_date = df["Join Date"].dropna().sort_values().iloc[
        len(df["Join Date"].dropna()) // 2
    ]

    # Fill missing values.
    df["Age"] = df["Age"].fillna(age_median)
    df["Salary"] = df["Salary"].fillna(salary_median)
    df["Country"] = df["Country"].fillna(country_mode)
    df["Name"] = df["Name"].fillna("Unknown")
    df["Join Date"] = df["Join Date"].fillna(median_date)

    # Calculate tenure using fixed date.
    reference_date = pd.Timestamp("2024-01-01")

    # Tenure means how many years the employee stayed in company.
    df["Tenure_Years"] = ((reference_date - df["Join Date"]).dt.days / 365.25).round(2)

    def flag_outliers(series):
        """
        Detect outliers using IQR method.

        Q1 = lower part of data. 25 percent values are below Q1.
        Q3 = higher part of data. 75 percent values are below Q3.
        IQR = Q3 - Q1. It shows spread of middle 50 percent data.

        Lower fence = minimum normal limit.
        Upper fence = maximum normal limit.

        If value is below lower fence or above upper fence,
        then it is treated as outlier.
        """

        # Q1 is 25% point of the data.
        q1 = series.quantile(0.25)

        # Q3 is 75% point of the data.
        q3 = series.quantile(0.75)

        # IQR is middle range of data.
        # Example: if Q1 = 27 and Q3 = 35, IQR = 8.
        iqr = q3 - q1

        # Lower fence formula.
        # Example: 27 - 1.5 * 8 = 15.
        lower = q1 - 1.5 * iqr

        # Upper fence formula.
        # Example: 35 + 1.5 * 8 = 47.
        upper = q3 + 1.5 * iqr

        # Return True if value is outside normal range.
        return ~series.between(lower, upper)

    # Add outlier columns.
    # True means this row is outlier.
    # False means this row is normal.
    df["Age_Outlier"] = flag_outliers(df["Age"])
    df["Salary_Outlier"] = flag_outliers(df["Salary"])

    # Sort by ID so output is clean.
    df = df.sort_values("ID").reset_index(drop=True)

    print("\nSTEP 2C — DATA WRANGLING COMPLETE")
    print(f"Age median used: {age_median}")
    print(f"Salary median used: {salary_median:,.0f}")
    print(f"Country mode used: {country_mode}")
    print(f"Median join date used: {median_date.date()}")

    return df


def clean_data(df):
    """Run full cleaning process."""
    df = data_scrubbing(df)
    df = data_munging(df)
    df = data_wrangling(df)

    print("\nCleaned Data:")
    print(df)

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df


def analyze_findings(clean_df):
    """Calculate main findings."""

    # Pearson correlation checks linear relationship.
    age_salary_corr = clean_df["Age"].corr(clean_df["Salary"], method="pearson")
    tenure_salary_corr = clean_df["Tenure_Years"].corr(clean_df["Salary"], method="pearson")

    # Slope shows how much salary changes for every 1 year age increase.
    slope, intercept = np.polyfit(clean_df["Age"], clean_df["Salary"], 1)

    under_30 = clean_df[clean_df["Age"] <= 30]
    under_30_average = under_30["Salary"].mean()

    grace = clean_df[clean_df["Name"] == "Grace"]

    print("\nSTEP 3 — TOP FINDINGS")
    print(f"1. Age vs Salary Pearson r: {age_salary_corr:.2f}")
    print(f"   Salary increases by around ${slope:,.0f} per year of age.")

    if not grace.empty:
        grace_salary = grace.iloc[0]["Salary"]
        grace_age = grace.iloc[0]["Age"]
        difference = grace_salary - under_30_average

        print(f"\n2. Grace is {grace_age:.0f} years old and earns ${grace_salary:,.0f}.")
        print(f"   This is ${difference:,.0f} above the under-30 average of ${under_30_average:,.0f}.")

    print(f"\n3. Tenure vs Salary Pearson r: {tenure_salary_corr:.2f}")
    print("   Tenure has almost no relationship with salary.")


def save_figure(output_dir, file_name, bg):
    """Save graph into output folder."""
    path = os.path.join(output_dir, file_name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close()
    print(f"Saved: {path}")


def plot_salary_distribution(clean_df, output_dir, bg):
    """Create salary distribution graph."""
    plt.figure(figsize=(9, 5))
    sns.histplot(clean_df["Salary"], bins=6, kde=True)

    plt.axvline(clean_df["Salary"].mean(), linestyle="--", label=f"Mean: ${clean_df['Salary'].mean():,.0f}")
    plt.axvline(clean_df["Salary"].median(), linestyle=":", label=f"Median: ${clean_df['Salary'].median():,.0f}")

    plt.title("Salary Distribution")
    plt.xlabel("Salary")
    plt.ylabel("Count")
    plt.legend()

    save_figure(output_dir, "salary_distribution.png", bg)


def plot_age_distribution(clean_df, output_dir, bg):
    """Create age distribution graph."""
    plt.figure(figsize=(9, 5))
    sns.histplot(clean_df["Age"], bins=6, kde=True)

    plt.axvline(clean_df["Age"].mean(), linestyle="--", label=f"Mean: {clean_df['Age'].mean():.1f}")
    plt.axvline(clean_df["Age"].median(), linestyle=":", label=f"Median: {clean_df['Age'].median():.1f}")

    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()

    save_figure(output_dir, "age_distribution.png", bg)


def plot_salary_by_country(clean_df, output_dir, bg):
    """Create average salary by country graph."""
    country_stats = clean_df.groupby("Country")["Salary"].agg(["mean", "count"]).reset_index()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=country_stats, x="Country", y="mean")

    # Add value labels on bars.
    for index, row in country_stats.iterrows():
        ax.text(index, row["mean"], f"${row['mean']:,.0f}\nn={row['count']}", ha="center", va="bottom")

    plt.title("Average Salary by Country")
    plt.xlabel("Country")
    plt.ylabel("Average Salary")

    save_figure(output_dir, "salary_by_country.png", bg)


def plot_age_vs_salary(clean_df, output_dir, bg):
    """Create Age vs Salary scatter plot."""
    plt.figure(figsize=(9, 5))
    sns.regplot(data=clean_df, x="Age", y="Salary")

    r = clean_df["Age"].corr(clean_df["Salary"], method="pearson")
    slope, intercept = np.polyfit(clean_df["Age"], clean_df["Salary"], 1)

    plt.title("Age vs Salary")
    plt.xlabel("Age")
    plt.ylabel("Salary")

    plt.text(
        0.05,
        0.92,
        f"Pearson r = {r:.2f}\nSlope = ${slope:,.0f}/year",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    save_figure(output_dir, "age_vs_salary_scatter.png", bg)


def plot_boxplots(clean_df, output_dir, bg):
    """Create boxplots for outlier checking."""
    plt.figure(figsize=(10, 5))

    # Boxplot uses Q1, Q3, IQR, lower fence, and upper fence.
    # It helps us see if there are outliers.
    sns.boxplot(data=clean_df[["Age", "Salary"]])

    plt.title("Boxplots for Outlier Inspection")

    save_figure(output_dir, "boxplots_outliers.png", bg)


def plot_correlation_heatmap(clean_df, output_dir, bg):
    """Create Pearson correlation heatmap."""

    # Only numeric columns can be used in Pearson correlation.
    numeric_cols = ["Age", "Salary", "Tenure_Years"]

    corr_matrix = clean_df[numeric_cols].corr(method="pearson")

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True
    )

    plt.title("Pearson Correlation Heatmap")

    save_figure(output_dir, "pearson_correlation_heatmap.png", bg)

    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(2))


def generate_graphs(clean_df, output_dir, bg):
    """Generate all required graphs."""
    print("\nSTEP 4 — GENERATE GRAPHS")

    sns.set_theme(style="whitegrid")

    plot_salary_distribution(clean_df, output_dir, bg)
    plot_age_distribution(clean_df, output_dir, bg)
    plot_salary_by_country(clean_df, output_dir, bg)
    plot_age_vs_salary(clean_df, output_dir, bg)
    plot_boxplots(clean_df, output_dir, bg)
    plot_correlation_heatmap(clean_df, output_dir, bg)


def report_outliers(clean_df):
    """Print outlier report using IQR method."""
    print("\nSTEP 5 — OUTLIER REPORT")

    for col in ["Age", "Salary"]:

        # Calculate quartiles
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)

        # IQR calculation
        iqr = q3 - q1

        # Lower and upper limits
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # Get outliers
        outliers = clean_df[(clean_df[col] < lower) | (clean_df[col] > upper)]

        # Print ONCE per column
        print(f"\nColumn: {col}")
        print(f"Q1: {q1:.1f}")
        print(f"Q3: {q3:.1f}")
        print(f"IQR: {iqr:.1f}")
        print(f"Lower fence: {lower:.1f}")
        print(f"Upper fence: {upper:.1f}")

        if outliers.empty:
            print("Result: No statistical outliers detected.")
        else:
            print("Outliers detected:")
            print(outliers[["ID", "Name", col]])


def main(file_path, output_dir, bg):
    """Run the full data cleaning and visualization pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_data(file_path)

    clean_df = clean_data(raw_df)

    analyze_findings(clean_df)

    generate_graphs(clean_df, output_dir, bg)

    report_outliers(clean_df)

    print(f"\nPipeline complete. All graphs saved in: {output_dir}")

    return clean_df


if __name__ == "__main__":
    # Main settings are here
    FILE_PATH = "messy_dataset.csv"
    OUTPUT_DIR = "graphs"
    BG = "#F8FAFC"

    main(FILE_PATH, OUTPUT_DIR, BG)