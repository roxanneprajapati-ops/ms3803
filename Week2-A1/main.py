import pandas as pd
import glob
from pathlib import Path


def load_and_combine_datasets(folder_path: str = "dataset") -> pd.DataFrame:
    """
    Load all CSV files from a folder and combine into one dataset.
    """
    # Create pattern like dataset/*.csv
    file_pattern = str(Path(folder_path) / "*.csv")

    # Get list of all CSV files
    file_list = glob.glob(file_pattern)

    # Check if folder is empty
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

    dataframes = []

    # Loop through each file
    for file in sorted(file_list):
        print(f"Loading file: {file}")  # show which file is loading
        df = pd.read_csv(file)          # read CSV file
        dataframes.append(df)           # store in list

    # Combine all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df


def describe_data_structure() -> None:
    """
    Print simple explanation of dataset structure.
    """
    print("\n=== Data Structure Description ===")
    print("Structured tabular data.")
    print("Each row = hourly air quality record.")
    print("Each column = variable (pollution, weather, time).")


def display_first_five_rows(df: pd.DataFrame) -> None:
    """
    Show first 5 rows of dataset.
    """
    print("\n=== First 5 Rows ===")
    print(df.head())  # display top rows


def display_column_names(df: pd.DataFrame) -> None:
    """
    Print all column names.
    """
    print("\n=== Column Names ===")
    for col in df.columns:
        print(col)  # print each column


def display_data_types(df: pd.DataFrame) -> None:
    """
    Show data types of each column.
    """
    print("\n=== Data Types ===")
    print(df.dtypes)  # show type (int, float, object)


def display_shape(df: pd.DataFrame) -> None:
    """
    Show total rows and columns.
    """
    rows, cols = df.shape  # get shape of dataset

    print("\n=== Dataset Size ===")
    print(f"Total rows: {rows}")
    print(f"Total columns: {cols}")


def main() -> None:
    """
    Main function to run Task 1.
    """
    try:
        print("Loading dataset from 'dataset/' folder...\n")

        # Load and combine all CSV files
        df = load_and_combine_datasets("dataset")

        print("\nDataset loaded successfully.")

        # Describe dataset
        describe_data_structure()

        # Show first rows
        display_first_five_rows(df)

        # Show column names
        display_column_names(df)

        # Show data types
        display_data_types(df)

        # Show dataset size
        display_shape(df)

    except FileNotFoundError as e:
        print(e)  # file not found error
    except Exception as e:
        print(f"An error occurred: {e}")  # general error


# Run the program
if __name__ == "__main__":
    main()