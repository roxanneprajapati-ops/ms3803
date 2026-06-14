"""
Data cleaning class for Avon River dataset.
This file handles loading Excel data, splitting water quality and fish
population data, cleaning both datasets, outlier checking, merging data,
saving cleaned CSV files.
"""

import os
import pandas as pd
import numpy as np


class DataCleaner:
    """
    DataCleaner prepares the raw Avon River dataset.
    This class only focus on data preparation. It does not run machine learning
    or create charts.
    """

    def __init__(self, input_file, output_dir):
        """
        Store input file and output folder.
        Also prepare variables that will be used across the cleaning process.
        """

        self.input_file = input_file
        self.output_dir = output_dir
        self.raw = None
        self.wq_raw = None
        self.fp_raw = None
        self.wq = None
        self.fp = None
        self.merged = None
        self.wq_stats = None
        self.fp_stats = None
        self.cleaning_log = []
        self.outlier_summary = []

        os.makedirs(self.output_dir, exist_ok=True)

    def print_step(self, title):
        """
        Print step title for console output.
        """

        print("\n" + "*" * 35)
        print(title)
        print("*" * 35)

    def load_raw_data(self):
        """
        Load raw Excel dataset.
        Header is in second row, so header=1 is used.
        """

        self.print_step("STEP 1: LOADING RAW DATA")
        self.raw = pd.read_excel(self.input_file, header=1)

        print("  Raw sheet loaded successfully")
        print(f"  Shape: {self.raw.shape[0]} rows x {self.raw.shape[1]} columns")
        print(f"  Columns: {self.raw.columns.tolist()}")

    def split_datasets(self):
        """
        Split raw Excel sheet into two datasets. The original sheet has water
        quality and fish population side by side.
        """

        self.print_step("STEP 2: SPLITTING DATASETS")

        self.wq_raw = self.raw[
            ["Site ID", "Date", "Temperature (°C)", "pH", "Dissolved Oxygen (mg/L)"]
        ].copy()

        self.wq_raw.columns = ["Site", "Date", "Temperature", "pH", "DO"]

        self.fp_raw = self.raw[
            ["Site ID.1", "Date.1", "Species", "Count", "Avg. Size (cm)"]
        ].copy()

        self.fp_raw.columns = ["Site", "Date", "Species", "Count", "AvgSize"]

        print(f"  Water Quality raw: {len(self.wq_raw)} rows")
        print(f"  Fish Population raw: {len(self.fp_raw)} rows")

    def clean_water_quality_data(self):
        """
        Clean water quality dataset.
        This method call smaller cleaning methods.
        """

        self.print_step("STEP 3: CLEANING WATER QUALITY DATA")

        self.wq = self.wq_raw.copy()
        self.convert_wq_dates()
        self.round_wq_values()
        self.impute_missing_ph()
        self.remove_wq_duplicates()
        self.resolve_wq_conflicts()

        print(f"\n  Water Quality clean: {len(self.wq)} rows")

    def convert_wq_dates(self):
        """
        Convert water quality dates to normal date format.
        """

        # excel date is converted to normal calendar date
        self.wq["Date"] = pd.to_datetime(self.wq["Date"])

        self.cleaning_log.append({
            "issue": "Excel serial date numbers",
            "rows": "All",
            "action": "Converted to standard calendar date format",
            "rows_lost": 0,
        })

        print("  3a. Dates converted")

    def round_wq_values(self):
        """
        Round water quality numbers.
        This remove floating point noise.
        """

        for col in ["Temperature", "pH", "DO"]:
            self.wq[col] = self.wq[col].round(2)

        self.cleaning_log.append({
            "issue": "Floating point precision noise",
            "rows": "Multiple",
            "action": "Rounded all numeric values to 2 decimal places",
            "rows_lost": 0,
        })

        print("  3b. Numeric values rounded")

    def impute_missing_ph(self):
        """
        Fill missing pH value using AV-3 mean.
        The missing value belongs to AV-3, so AV-3 site mean is used instead of
        full dataset mean.
        """

        av3_ph_mean = round(self.wq[self.wq["Site"] == "AV-3"]["pH"].mean(), 2)
        missing_idx = self.wq[self.wq["pH"].isnull()].index.tolist()

        self.wq.loc[missing_idx, "pH"] = av3_ph_mean

        self.cleaning_log.append({
            "issue": "Missing pH value",
            "rows": len(missing_idx),
            "action": f"Imputed with AV-3 site mean ({av3_ph_mean})",
            "rows_lost": 0,
        })

        print(f"  3c. Missing pH filled: {len(missing_idx)} row(s)")

    def remove_wq_duplicates(self):
        """
        Remove exact duplicate water quality records.
        """

        before = len(self.wq)
        self.wq = self.wq.drop_duplicates()
        lost = before - len(self.wq)

        if lost:
            self.cleaning_log.append({
                "issue": "Exact duplicate rows (Water Quality)",
                "rows": lost,
                "action": "Removed exact duplicates",
                "rows_lost": lost,
            })

        print(f"  3d. Exact duplicates removed: {lost}")

    def resolve_wq_conflicts(self):
        """
        Resolve conflicting water quality records.

        Same site and date but different values:
        - average Temperature
        - average DO
        - keep lower pH for conservative risk view
        """

        conflict = self.wq[self.wq.duplicated(subset=["Site", "Date"], keep=False)]

        if len(conflict):
            print(f"  3e. Conflicting records found: {len(conflict)}")

            self.wq = (
                self.wq.groupby(["Site", "Date"], as_index=False)
                .agg({
                    "Temperature": "mean",
                    "pH": "min",
                    "DO": "mean",
                })
            )

            for col in ["Temperature", "pH", "DO"]:
                self.wq[col] = self.wq[col].round(2)

            self.cleaning_log.append({
                "issue": "Conflicting duplicate water quality record",
                "rows": len(conflict),
                "action": "Averaged Temperature/DO and kept lower pH",
                "rows_lost": len(conflict) - 1,
            })

            print("      Conflict resolved")
        else:
            print("  3e. No conflicting water quality records")

    def clean_fish_population_data(self):
        """
        Clean fish population dataset.
        This method call smaller fish cleaning methods.
        """

        self.print_step("STEP 4: CLEANING FISH POPULATION DATA")

        self.fp = self.fp_raw.copy()

        self.convert_fp_dates()
        self.round_fp_values()
        self.fix_missing_species()
        self.remove_fp_duplicates()
        self.check_multi_species_records()

        print(f"\n  Fish Population clean: {len(self.fp)} rows")

    def convert_fp_dates(self):
        """
        Convert fish population dates.
        """

        self.fp["Date"] = pd.to_datetime(self.fp["Date"])
        print("  4a. Dates converted")

    def round_fp_values(self):
        """
        Round average fish size values.
        """

        self.fp["AvgSize"] = self.fp["AvgSize"].round(2)
        print("  4b. Average size rounded")

    def fix_missing_species(self):
        """
        Fill missing species label.
        Count and size are still valid, so row is kept.
        """

        missing_idx = self.fp[self.fp["Species"].isnull()].index.tolist()

        if missing_idx:
            self.fp.loc[missing_idx, "Species"] = "Unknown (possible Inanga)"

            self.cleaning_log.append({
                "issue": "Missing Species value",
                "rows": len(missing_idx),
                "action": "Flagged as Unknown (possible Inanga)",
                "rows_lost": 0,
            })

        print(f"  4c. Missing species fixed: {len(missing_idx)} row(s)")

    def remove_fp_duplicates(self):
        """
        Remove exact duplicate fish population records.
        """

        before = len(self.fp)
        self.fp = self.fp.drop_duplicates()
        lost = before - len(self.fp)

        if lost:
            self.cleaning_log.append({
                "issue": "Exact duplicate rows (Fish Population)",
                "rows": lost,
                "action": "Removed exact duplicates",
                "rows_lost": lost,
            })

        print(f"  4d. Exact duplicates removed: {lost}")

    def check_multi_species_records(self):
        """
        Check same-site same-date fish records.
        Different species on same day are valid, so these records are kept.
        """

        conflict = self.fp[self.fp.duplicated(subset=["Site", "Date"], keep=False)]

        if len(conflict):
            self.cleaning_log.append({
                "issue": "Same-date multi-species records",
                "rows": len(conflict),
                "action": "Kept both because different species are valid",
                "rows_lost": 0,
            })

        print(f"  4e. Multi-species same-date records kept: {len(conflict)}")

    def detect_outliers(self):
        """
        Detect outliers using IQR and Z-score.
        No values are removed. Important values are only flagged.
        """

        self.print_step("STEP 5: OUTLIER DETECTION")
        self.detect_iqr_outliers()
        self.detect_zscore_outliers()
        self.detect_site_do_outliers()
        self.print_ecological_outlier_notes()

    def detect_iqr_outliers(self):
        """
        Detect outliers using IQR method.
        """

        wq_cols = {
            "Temperature": "Temperature (°C)",
            "pH": "pH",
            "DO": "Dissolved Oxygen (mg/L)",
        }

        fp_cols = {
            "Count": "Fish Count",
            "AvgSize": "Average Size (cm)",
        }

        print("\n  Method 1 — IQR Method:")

        for col, label in wq_cols.items():
            self.add_iqr_summary(self.wq, col, label)

        for col, label in fp_cols.items():
            df = self.fp[self.fp[col].notna()]
            self.add_iqr_summary(df, col, label)

    def add_iqr_summary(self, df, col, label):
        """
        Calculate and store IQR outlier summary.
        """

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outs = df[(df[col] < lower) | (df[col] > upper)]

        self.outlier_summary.append({
            "variable": label,
            "mean": round(df[col].mean(), 2),
            "min": round(df[col].min(), 2),
            "max": round(df[col].max(), 2),
            "lower_fence": round(lower, 2),
            "upper_fence": round(upper, 2),
            "outliers": len(outs),
        })

        status = "NO OUTLIERS" if not len(outs) else f"{len(outs)} OUTLIER(S)"
        print(f"  {label}: {status} | fence [{lower:.2f}, {upper:.2f}]")

    def detect_zscore_outliers(self):
        """
        Detect outliers using Z-score method.
        """

        print("\n  Method 2 — Z-Score Method:")

        cols = {
            "Temperature": ("Temperature (°C)", self.wq),
            "pH": ("pH", self.wq),
            "DO": ("Dissolved Oxygen (mg/L)", self.wq),
            "Count": ("Fish Count", self.fp),
            "AvgSize": ("Average Size (cm)", self.fp),
        }

        for col, (label, df) in cols.items():
            values = df[col].dropna()
            z = np.abs((values - values.mean()) / values.std())
            z_outs = values[z > 3]

            print(f"  {label}: {len(z_outs)} Z-score outlier(s)")

    def detect_site_do_outliers(self):
        """
        Check DO outliers per site.
        DO is most important for fish survival.
        """

        print("\n  Per-site DO check:")

        for site in ["AV-1", "AV-2", "AV-3"]:
            d = self.wq[self.wq["Site"] == site]["DO"]

            q1 = d.quantile(0.25)
            q3 = d.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outs = d[(d < lower) | (d > upper)]

            print(f"  {site}: {len(outs)} DO outlier(s)")

    def print_ecological_outlier_notes(self):
        """
        Print ecological notes.
        Some values may not be statistical outliers, but still important for river health.
        """

        min_do_av3 = self.wq[self.wq["Site"] == "AV-3"]["DO"].min()
        size_min = self.fp["AvgSize"].min()
        size_max = self.fp["AvgSize"].max()

        print("\n  Ecological Notes:")
        print(f"  AV-3 min DO is {min_do_av3} mg/L, below 7.0 mg/L threshold.")
        print(f"  Fish size range is {size_min} to {size_max} cm.")
        print("  All records retained. No values removed on outlier grounds.")

    def merge_datasets(self):
        """
        Merge water quality and fish population datasets.
        """

        self.print_step("STEP 6: MERGING DATASETS")

        self.merged = pd.merge(
            self.wq,
            self.fp,
            on=["Site", "Date"],
            how="inner",
        )

        self.calculate_summary_stats()

        print(f"  Merged dataset: {len(self.merged)} rows")

    def calculate_summary_stats(self):
        """
        Calculate summary statistics by site.
        """

        self.wq_stats = self.wq.groupby("Site").agg(
            Temp_mean=("Temperature", "mean"),
            Temp_min=("Temperature", "min"),
            Temp_max=("Temperature", "max"),
            pH_mean=("pH", "mean"),
            pH_min=("pH", "min"),
            pH_max=("pH", "max"),
            DO_mean=("DO", "mean"),
            DO_min=("DO", "min"),
            DO_max=("DO", "max"),
            obs=("DO", "count"),
        ).round(2)

        self.fp_stats = self.fp.groupby("Site").agg(
            count_mean=("Count", "mean"),
            count_sum=("Count", "sum"),
            size_mean=("AvgSize", "mean"),
            species_n=("Species", "nunique"),
        ).round(2)

    def save_cleaned_datasets(self):
        """
        Save cleaned datasets to CSV.
        """

        self.print_step("SAVING CLEANED DATASETS")

        self.wq.to_csv(
            os.path.join(self.output_dir, "water_quality_clean.csv"),
            index=False,
        )

        self.fp.to_csv(
            os.path.join(self.output_dir, "fish_population_clean.csv"),
            index=False,
        )

        self.merged.to_csv(
            os.path.join(self.output_dir, "avon_river_merged_clean.csv"),
            index=False,
        )

        print("  Cleaned CSV files saved")