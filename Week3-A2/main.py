import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")


def load_data(file_path):
    """Load CSV file and show first data check."""
    df = pd.read_csv(file_path)

    print("\nSTEP 1 — LOAD DATA")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nRaw Data:")
    print(df)

    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    return df


def data_scrubbing(df):
    """
    Data Scrubbing: fix dirty values.

    Cleaned:
    - Age text to number
    - Salary text to number
    - Duplicate ID rows
    - Invalid Join Date
    """
    df = df.copy()

    # Word to number map
    word_to_int = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }

    def words_to_number(value):
        """Convert text age to number."""
        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower()

        try:
            return float(value)
        except ValueError:
            total = 0

            # Example: thirty-eight = 30 + 8
            for word in value.replace("-", " ").split():
                total += word_to_int.get(word, 0)

            return float(total) if total > 0 else np.nan

    def clean_salary(value):
        """Convert salary to number."""
        if pd.isna(value):
            return np.nan

        value = str(value).strip().lower().replace(",", "").replace("$", "")

        try:
            return float(value)
        except ValueError:
            total = 0
            multiplier = 1

            # Example: sixty five thousand = 65 * 1000
            for word in value.split():
                if word == "thousand":
                    multiplier = 1000
                else:
                    total += word_to_int.get(word, 0)

            return float(total * multiplier) if total > 0 else np.nan

    # Clean Age and Salary
    df["Age"] = df["Age"].apply(words_to_number)
    df["Salary"] = df["Salary"].apply(clean_salary)

    # Convert ID to number
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce")

    # Find duplicate IDs
    duplicated_ids = df[df.duplicated("ID", keep=False) & df["ID"].notna()]["ID"].unique()

    merged_rows = []

    # Merge duplicate rows
    for duplicate_id in duplicated_ids:
        group = df[df["ID"] == duplicate_id]
        merged = {}

        for col in group.columns:
            values = group[col].dropna()
            merged[col] = values.iloc[0] if len(values) > 0 else np.nan

        merged_rows.append(merged)

    # Remove old duplicate rows
    df = df.drop(df[df["ID"].isin(duplicated_ids)].index)

    # Add merged rows
    if merged_rows:
        df = pd.concat([df, pd.DataFrame(merged_rows)], ignore_index=True)

    # Convert date
    df["Join Date"] = pd.to_datetime(df["Join Date"], errors="coerce", dayfirst=True)

    print("\nSTEP 2A — DATA SCRUBBING DONE")
    return df


def data_munging(df):
    """
    Data Munging: fix data format.

    Cleaned:
    - Country codes
    - Name format
    - Numeric data types
    """
    df = df.copy()

    # Country code map
    country_map = {
        "AU": "AUS",
        "AUS": "AUS",
        "NZ": "NZL",
        "NZL": "NZL"
    }

    # Clean country
    df["Country"] = df["Country"].astype(str).str.strip().str.upper().map(country_map)

    # Clean name
    df["Name"] = df["Name"].astype(str).str.strip().str.title()
    df["Name"] = df["Name"].replace("Nan", np.nan)

    # Fix data types
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

    print("\nSTEP 2B — DATA MUNGING DONE")
    return df


def data_wrangling(df):
    """
    Data Wrangling: prepare final data.

    Prepared:
    - Fill missing values
    - Convert AUD salary to NZD
    - Create Tenure_Years
    - Add outlier flags
    """
    df = df.copy()

    # Fill missing values
    age_median = df["Age"].median()
    salary_median = df["Salary"].median()
    country_mode = df["Country"].mode()[0]

    median_date = df["Join Date"].dropna().sort_values().iloc[
        len(df["Join Date"].dropna()) // 2
    ]

    df["Age"] = df["Age"].fillna(age_median)
    df["Salary"] = df["Salary"].fillna(salary_median)
    df["Country"] = df["Country"].fillna(country_mode)
    df["Name"] = df["Name"].fillna("Unknown")
    df["Join Date"] = df["Join Date"].fillna(median_date)

    # Convert AUD to NZD
    # Original Salary stays the same
    # Salary_NZD is used for analysis
    aud_to_nzd = 1.08
    df["Salary_NZD"] = df["Salary"]

    df.loc[df["Country"] == "AUS", "Salary_NZD"] = (
        df["Salary"] * aud_to_nzd
    )

    # Create tenure
    reference_date = pd.Timestamp("2024-01-01")
    df["Tenure_Years"] = ((reference_date - df["Join Date"]).dt.days / 365.25).round(2)

    def flag_outliers(series):
        """
        Find outliers using IQR.

        Q1 = 25% point
        Q3 = 75% point
        IQR = Q3 - Q1
        Outside lower/upper fence = outlier
        """
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return ~series.between(lower, upper)

    # Outlier flags
    df["Age_Outlier"] = flag_outliers(df["Age"])
    df["Salary_NZD_Outlier"] = flag_outliers(df["Salary_NZD"])

    # Sort by ID
    df = df.sort_values("ID").reset_index(drop=True)

    print("\nSTEP 2C — DATA WRANGLING DONE")
    print(f"Age median used: {age_median}")
    print(f"Salary median used: {salary_median:,.0f}")
    print(f"Country mode used: {country_mode}")
    print(f"Median join date used: {median_date.date()}")
    print("Currency conversion: AUS salary * 1.08 = Salary_NZD")

    return df


def clean_data(df):
    """
    Run all cleaning steps.

    Steps:
    1. Scrubbing
    2. Munging
    3. Wrangling
    """
    df = data_scrubbing(df)
    df = data_munging(df)
    df = data_wrangling(df)

    print("\nCleaned Data:")
    print(df)

    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())

    return df


def analyze_findings(clean_df):
    """
    Analyze cleaned data.

    Checked:
    - Age vs Salary_NZD
    - Tenure vs Salary_NZD
    - Salary increase per age year
    """
    age_salary_corr = clean_df["Age"].corr(clean_df["Salary_NZD"], method="pearson")
    tenure_salary_corr = clean_df["Tenure_Years"].corr(clean_df["Salary_NZD"], method="pearson")

    slope, intercept = np.polyfit(clean_df["Age"], clean_df["Salary_NZD"], 1)

    under_30 = clean_df[clean_df["Age"] <= 30]
    under_30_average = under_30["Salary_NZD"].mean()

    grace = clean_df[clean_df["Name"] == "Grace"]

    print("\nSTEP 3 — TOP FINDINGS")
    print(f"1. Age vs Salary_NZD Pearson r: {age_salary_corr:.2f}")
    print(f"   Salary increases by around ${slope:,.0f} NZD per year of age.")

    if not grace.empty:
        grace_salary = grace.iloc[0]["Salary_NZD"]
        grace_age = grace.iloc[0]["Age"]
        difference = grace_salary - under_30_average

        print(f"\n2. Grace is {grace_age:.0f} years old and earns ${grace_salary:,.0f} NZD.")
        print(f"   This is ${difference:,.0f} above the under-30 average of ${under_30_average:,.0f} NZD.")

    print(f"\n3. Tenure vs Salary_NZD Pearson r: {tenure_salary_corr:.2f}")
    print("   Tenure has almost no relationship with salary.")

    return age_salary_corr, tenure_salary_corr, slope


def save_figure(output_dir, file_name, bg):
    """Save graph."""
    path = os.path.join(output_dir, file_name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close()
    print(f"Saved: {path}")


def plot_salary_distribution(clean_df, output_dir, bg):
    """Show salary spread."""
    plt.figure(figsize=(9, 5))
    sns.histplot(clean_df["Salary_NZD"], bins=6, kde=True)

    plt.axvline(
        clean_df["Salary_NZD"].mean(),
        linestyle="--",
        label=f"Mean: ${clean_df['Salary_NZD'].mean():,.0f}"
    )

    plt.axvline(
        clean_df["Salary_NZD"].median(),
        linestyle=":",
        label=f"Median: ${clean_df['Salary_NZD'].median():,.0f}"
    )

    plt.title("Salary Distribution (NZD)")
    plt.xlabel("Salary (NZD)")
    plt.ylabel("Count")
    plt.legend()

    save_figure(output_dir, "salary_distribution.png", bg)


def plot_age_distribution(clean_df, output_dir, bg):
    """Show age spread."""
    plt.figure(figsize=(9, 5))
    sns.histplot(clean_df["Age"], bins=6, kde=True)

    plt.axvline(
        clean_df["Age"].mean(),
        linestyle="--",
        label=f"Mean: {clean_df['Age'].mean():.1f}"
    )

    plt.axvline(
        clean_df["Age"].median(),
        linestyle=":",
        label=f"Median: {clean_df['Age'].median():.1f}"
    )

    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.legend()

    save_figure(output_dir, "age_distribution.png", bg)


def plot_salary_by_country(clean_df, output_dir, bg):
    """Compare salary by country."""
    country_stats = clean_df.groupby("Country")["Salary_NZD"].agg(["mean", "count"]).reset_index()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=country_stats, x="Country", y="mean")

    for index, row in country_stats.iterrows():
        ax.text(
            index,
            row["mean"],
            f"${row['mean']:,.0f}\nn={row['count']}",
            ha="center",
            va="bottom"
        )

    plt.title("Average Salary by Country (NZD)")
    plt.xlabel("Country")
    plt.ylabel("Average Salary (NZD)")

    save_figure(output_dir, "salary_by_country.png", bg)


def plot_age_vs_salary(clean_df, output_dir, bg):
    """Show age and salary relationship."""
    plt.figure(figsize=(9, 5))
    sns.regplot(data=clean_df, x="Age", y="Salary_NZD")

    r = clean_df["Age"].corr(clean_df["Salary_NZD"], method="pearson")
    slope, intercept = np.polyfit(clean_df["Age"], clean_df["Salary_NZD"], 1)

    plt.title("Age vs Salary (NZD)")
    plt.xlabel("Age")
    plt.ylabel("Salary (NZD)")

    plt.text(
        0.05,
        0.92,
        f"Pearson r = {r:.2f}\nSlope = ${slope:,.0f}/year",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8)
    )

    save_figure(output_dir, "age_vs_salary_scatter.png", bg)


def plot_age_vs_salary_detailed(clean_df, output_dir, bg):
    """Show detailed age and salary graph."""
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.set_facecolor(bg)

    country_colour = {
        "NZL": "#065A82",
        "AUS": "#00A896"
    }

    colours = clean_df["Country"].map(country_colour).fillna("#94A3B8")

    ax.scatter(
        clean_df["Age"],
        clean_df["Salary_NZD"],
        c=colours,
        s=100,
        edgecolors="white",
        linewidth=1,
        alpha=0.95
    )

    # Add employee names
    for _, row in clean_df.iterrows():
        ax.annotate(
            str(row["Name"]),
            (row["Age"], row["Salary_NZD"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color="#475569"
        )

    r = clean_df["Age"].corr(clean_df["Salary_NZD"], method="pearson")
    slope, intercept = np.polyfit(clean_df["Age"], clean_df["Salary_NZD"], 1)

    x_line = np.linspace(clean_df["Age"].min(), clean_df["Age"].max(), 100)
    y_line = slope * x_line + intercept

    ax.plot(
        x_line,
        y_line,
        color="#F96167",
        linestyle="--",
        linewidth=2.5,
        label=f"Trend (slope ${slope:,.0f}/yr)"
    )

    ax.text(
        0.05,
        0.92,
        f"Pearson r = {r:.2f}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color="#F96167",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#CBD5E1",
            alpha=0.95
        )
    )

    ax.set_ylim(50000, 80000)
    ax.set_yticks([50000, 55000, 60000, 65000, 70000, 75000, 80000])

    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

    legend_handles = [
        mpatches.Patch(color="#065A82", label="NZL"),
        mpatches.Patch(color="#00A896", label="AUS"),
        plt.Line2D(
            [0],
            [0],
            color="#F96167",
            linestyle="--",
            linewidth=2,
            label=f"Trend (slope ${slope:,.0f}/yr)"
        )
    ]

    ax.legend(handles=legend_handles, frameon=False, loc="lower right")

    ax.set_title(
        "Age vs Salary — Scatter Plot with Regression Line (NZD)",
        fontsize=15,
        fontweight="bold",
        color="#1E293B"
    )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Salary (NZD)")
    ax.grid(alpha=0.3)

    save_figure(output_dir, "age_vs_salary_detailed.png", bg)


def plot_boxplots(clean_df, output_dir, bg):
    """Show possible outliers."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=clean_df[["Age", "Salary_NZD"]])

    plt.title("Boxplots for Outlier Inspection")

    save_figure(output_dir, "boxplots_outliers.png", bg)


def plot_correlation_heatmap(clean_df, output_dir, bg):
    """Show Pearson correlation."""
    numeric_cols = ["Age", "Salary_NZD", "Tenure_Years"]
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
    """Generate all graphs."""
    print("\nSTEP 4 — GENERATE GRAPHS")

    sns.set_theme(style="whitegrid")

    plot_salary_distribution(clean_df, output_dir, bg)
    plot_age_distribution(clean_df, output_dir, bg)
    plot_salary_by_country(clean_df, output_dir, bg)
    plot_age_vs_salary(clean_df, output_dir, bg)
    plot_age_vs_salary_detailed(clean_df, output_dir, bg)
    plot_boxplots(clean_df, output_dir, bg)
    plot_correlation_heatmap(clean_df, output_dir, bg)


def report_outliers(clean_df):
    """
    Show statistical outliers.

    Checked:
    - Age
    - Salary_NZD
    """
    print("\nSTEP 5 — OUTLIER REPORT")

    for col in ["Age", "Salary_NZD"]:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = clean_df[(clean_df[col] < lower) | (clean_df[col] > upper)]

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
    """Run full pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    raw_df = load_data(file_path)
    clean_df = clean_data(raw_df)

    analyze_findings(clean_df)
    generate_graphs(clean_df, output_dir, bg)
    report_outliers(clean_df)

    print(f"\nPipeline complete. All graphs saved in: {output_dir}")

    return clean_df


if __name__ == "__main__":
    FILE_PATH = "messy_dataset.csv"
    OUTPUT_DIR = "graphs"
    BG = "#F8FAFC"

    main(FILE_PATH, OUTPUT_DIR, BG)