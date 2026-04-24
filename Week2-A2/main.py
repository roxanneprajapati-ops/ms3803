# -----------------------------------------------------------------------------
# main.py - Beijing Multi-Site Air Quality Analysis
# Author: Roxanne Prajapati
# Description:
#       This program analyzes the Beijing Multi-Site Air Quality dataset by loading
# several CSV files and combining them into one dataset. It first checks the
# structure of the data, then cleans it by handling missing values and removing
# duplicates. After that, it calculates basic statistics and filters the data to
# look at pollution levels for each station. Finally, it creates some graphs and
# checks the relationship between variables, with the results shown in the
# console and saved in the outputs folder.
# -----------------------------------------------------------------------------

import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt


POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]


def print_table(df: pd.DataFrame) -> None:
    """
    Print dataframe as clean aligned table.
    """
    df = df.copy()

    # format numeric values to 2 decimal places
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:,.2f}")

    index_name = df.index.name if df.index.name else ""

    # get width for index and columns
    index_width = max(len(str(index_name)), max(len(str(i)) for i in df.index))
    col_widths = {}

    for col in df.columns:
        col_widths[col] = max(len(str(col)), max(len(str(v)) for v in df[col]))

    # print header
    header = f"| {index_name:<{index_width}} "
    for col in df.columns:
        header += f"| {col:>{col_widths[col]}} "
    header += "|"

    # print separator
    separator = f"|{'-' * (index_width + 2)}"
    for col in df.columns:
        separator += f"|{'-' * (col_widths[col] + 2)}"
    separator += "|"

    print(header)
    print(separator)

    # print rows
    for idx, row in df.iterrows():
        line = f"| {str(idx):<{index_width}} "
        for col in df.columns:
            line += f"| {str(row[col]):>{col_widths[col]}} "
        line += "|"
        print(line)


def load_and_combine_datasets(folder_path: str = "dataset") -> pd.DataFrame:
    """
    Load all CSV files from a folder and combine into one dataset.
    """
    # create pattern like dataset/*.csv
    file_pattern = str(Path(folder_path) / "*.csv")

    # get list of all CSV files
    file_list = glob.glob(file_pattern)

    # check if folder is empty
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    dataframes = []

    # loop through each file
    for file in sorted(file_list):
        print(f"Loading file: {file}")  # show which file is loading
        df = pd.read_csv(file)          # read CSV file
        dataframes.append(df)           # store in list

    # combine all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def data_structure(df: pd.DataFrame) -> None:
    """
    TASK 1: show basic structure of dataset.
    """
    print("\n" + "═" * 60)
    print("TASK 1 – Data Structure")
    print("═" * 60)

    print("\n=== Data Structure Description ===")
    print("Structured tabular data.")
    print("Each row = hourly air quality record.")
    print("Each column = variable such as pollution, weather, time, or station.")

    print("\n=== First 5 Rows ===")
    print(df.head())  # show first 5 rows

    print("\n=== Column Names ===")
    for col in df.columns:
        print(col)  # print each column

    print("\n=== Data Types ===")
    dtype_table = pd.DataFrame({"Data Type": df.dtypes.astype(str)})
    print_table(dtype_table)

    rows, cols = df.shape
    print("\n=== Dataset Size ===")
    print(f"Total rows: {rows:,}")
    print(f"Total columns: {cols}")


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 2: clean data using time-based fill.
    Missing values are fill using nearby value in same station.
    """
    print("\n" + "═" * 60)
    print("TASK 2 – Data Cleaning")
    print("═" * 60)

    # make copy so original data is not directly changed
    df = df.copy()

    # check missing values count and percent
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    # create missing value summary table
    audit = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct
    })

    # show only columns with missing value
    audit = audit[audit["Missing Count"] > 0].sort_values(
        "Missing %",
        ascending=False
    )

    if audit.empty:
        print("No missing values detected.")
    else:
        print("\n── Missing-value summary ──")
        print_table(audit)

    # create datetime column from year, month, day, hour
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

    # sort by station and time before filling missing value
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # get numeric columns only
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # fill missing numeric values using forward fill and backfill per station
    for col in numeric_cols:
        df[col] = (
            df.groupby("station")[col]
              .transform(lambda s: s.ffill().bfill())
        )

    # check if there are still missing values
    remaining_null = df.isnull().sum().sum()
    print(f"\nRemaining null values after fill: {remaining_null:,}")

    # remove rows if key pollutants are still missing
    before = len(df)
    df = df.dropna(subset=POLLUTANTS)
    after = len(df)

    print(f"Rows dropped with still-null pollutants: {before - after:,}")

    # remove duplicate rows
    before_dup = len(df)
    df = df.drop_duplicates()
    after_dup = len(df)

    print(f"Duplicate rows removed: {before_dup - after_dup:,}")
    print(f"Final clean dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df.reset_index(drop=True)


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 3: calculate basic statistics for pollutant columns.
    """
    print("\n" + "═" * 60)
    print("TASK 3 – Basic Statistical Analysis")
    print("═" * 60)

    # calculate mean, median, min, max and standard deviation
    stats = df[POLLUTANTS].agg(["mean", "median", "min", "max", "std"]).T
    stats.columns = ["Mean", "Median", "Min", "Max", "Std Dev"]
    stats = stats.round(2)

    print("\n── Descriptive statistics (all stations combined) ──")
    print_table(stats)

    # calculate PM2.5 statistics per station
    print("\n── PM2.5 statistics per station ──")
    station_stats = (
        df.groupby("station")["PM2.5"]
          .agg(["mean", "median", "min", "max", "std"])
          .rename(columns={
              "mean": "Mean",
              "median": "Median",
              "min": "Min",
              "max": "Max",
              "std": "Std Dev"
          })
          .round(2)
    )

    print_table(station_stats)

    # simple finding from statistics
    highest_station = station_stats["Mean"].idxmax()
    lowest_station = station_stats["Mean"].idxmin()

    print("\n── Simple findings ──")
    print(f"Highest average PM2.5 station: {highest_station}")
    print(f"Lowest average PM2.5 station: {lowest_station}")
    print(f"Overall average PM2.5: {stats.loc['PM2.5', 'Mean']}")

    return stats


def data_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 4: group by station and compare average pollution.
    """
    print("\n" + "═" * 60)
    print("TASK 4 – Data Filtering")
    print("═" * 60)

    # calculate average pollution for each station
    avg_pollution = (
        df.groupby("station")[POLLUTANTS]
          .mean()
          .round(2)
    )

    print("\n── Average pollutant concentrations per station ──")
    print_table(avg_pollution)

    # find highest and lowest average PM2.5 station
    worst = avg_pollution["PM2.5"].idxmax()
    best = avg_pollution["PM2.5"].idxmin()

    print(f"\nHighest average PM2.5: {worst} ({avg_pollution.loc[worst, 'PM2.5']:.1f})")
    print(f"Lowest average PM2.5: {best} ({avg_pollution.loc[best, 'PM2.5']:.1f})")

    # count hours where PM2.5 is very high
    hazard = (
        df[df["PM2.5"] > 150]
        .groupby("station")
        .size()
        .rename("Hazardous hours")
        .to_frame()
    )

    print("\n── Hours exceeding PM2.5 > 150 per station ──")
    print_table(hazard)

    return avg_pollution


def data_visualization(df: pd.DataFrame) -> None:
    """
    TASK 5: create histogram, line plot and boxplot.
    """
    print("\n" + "═" * 60)
    print("TASK 5 – Data Visualization")
    print("═" * 60)

    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)  # create outputs folder if not exist

    # histogram of PM2.5
    plt.figure(figsize=(8, 5))
    plt.hist(df["PM2.5"], bins=40)
    plt.title("PM2.5 Distribution")
    plt.xlabel("PM2.5")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_folder / "hist.png")
    plt.close()

    # line plot of monthly average PM2.5
    df["year_month"] = df["datetime"].dt.to_period("M")
    monthly_pm25 = df.groupby("year_month")["PM2.5"].mean()
    monthly_pm25.index = monthly_pm25.index.to_timestamp()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_pm25.index, monthly_pm25.values)
    plt.title("Monthly Average PM2.5 Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average PM2.5")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_folder / "line.png")
    plt.close()

    # boxplot of pollutant columns
    plt.figure(figsize=(9, 5))
    df[POLLUTANTS].boxplot()
    plt.title("Boxplot of Pollutants")
    plt.ylabel("Pollution Level")
    plt.tight_layout()
    plt.savefig(output_folder / "box.png")
    plt.close()

    print("\nCharts saved inside outputs folder:")
    print("- outputs/hist.png")
    print("- outputs/line.png")
    print("- outputs/box.png")

    # scatter plots to show relationship with PM2.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(df["PM10"], df["PM2.5"], alpha=0.2)
    axes[0].set_title("PM10 vs PM2.5")

    axes[1].scatter(df["NO2"], df["PM2.5"], alpha=0.2)
    axes[1].set_title("NO2 vs PM2.5")

    axes[2].scatter(df["CO"], df["PM2.5"], alpha=0.2)
    axes[2].set_title("CO vs PM2.5")

    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("PM2.5")

    plt.tight_layout()
    plt.savefig(output_folder / "pollutants_vs_pm25.png")
    plt.close()

    # bar chart for average PM2.5 per station
    avg_pm25 = (
        df.groupby("station")["PM2.5"]
        .mean()
        .sort_values(ascending=False)  # sort highest to lowest
    )

    plt.figure(figsize=(10, 5))
    avg_pm25.plot(kind="bar")

    plt.title("Average PM2.5 by Station")
    plt.xlabel("Station")
    plt.ylabel("Average PM2.5")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_folder / "avg_pm25_by_station.png")
    plt.close()


def correlation_analysis(df: pd.DataFrame) -> None:
    """
    TASK 6: check which variables are related to PM2.5.
    """
    print("\n" + "═" * 60)
    print("TASK 6 – Correlation Analysis")
    print("═" * 60)

    # use pollutant and weather columns
    columns = POLLUTANTS + ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    available_columns = [col for col in columns if col in df.columns]

    # calculate correlation with PM2.5
    numeric_df = df[available_columns].dropna()
    correlation = numeric_df.corr()["PM2.5"].drop("PM2.5")
    correlation = correlation.sort_values(key=abs, ascending=False)

    correlation_table = correlation.round(3).to_frame(name="Correlation")

    print("\n── Correlation with PM2.5 ──")
    print_table(correlation_table)

    # find strongest related variable
    strongest_variable = correlation.abs().idxmax()
    strongest_value = correlation[strongest_variable]

    print("\n── Most related variable ──")
    print(f"{strongest_variable}: {strongest_value:.3f}")

    # check temperature relationship
    if "TEMP" in correlation.index:
        temp_corr = correlation["TEMP"]

        print("\n── Temperature and PM2.5 ──")
        print(f"TEMP correlation with PM2.5: {temp_corr:.3f}")

        if abs(temp_corr) < 0.2:
            print("Interpretation: Temperature has weak relationship with PM2.5.")
        elif temp_corr > 0:
            print("Interpretation: Temperature has positive relationship with PM2.5.")
        else:
            print("Interpretation: Temperature has negative relationship with PM2.5.")


def main() -> None:
    """
    Main function to run all task.
    """
    try:
        print("Loading dataset from 'dataset/' folder...\n")

        # load and combine all CSV files
        df = load_and_combine_datasets("dataset")

        print("\nDataset loaded successfully.")

        # run task 1
        data_structure(df)

        # run task 2 and use clean dataset after this
        cleaned_df = data_cleaning(df)

        # run task 3
        basic_statistics(cleaned_df)

        # run task 4
        data_filtering(cleaned_df)

        # run task 5
        data_visualization(cleaned_df)

        # run task 6
        correlation_analysis(cleaned_df)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


# run the program
if __name__ == "__main__":
    main()