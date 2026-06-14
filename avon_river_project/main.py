"""
Avon River Data Analytics Pipeline : This file is the main runner.
It connects data cleaning, machine learning, chart generation and Power BI export.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from data_cleaner import DataCleaner
from ml_analyzer import MLAnalyzer
from chart_generator import ChartGenerator
from dashboard_exporter import PowerBIExporter


INPUT_FILE = "data/dataset.xlsx"
OUTPUT_DIR = "output"
CLEANED_DATA_DIR = os.path.join(OUTPUT_DIR, "cleaned_data")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
POWERBI_DIR = os.path.join(OUTPUT_DIR, "powerbi")


def create_output_folders():
    """
    Create all output folders used by the pipeline.
    """

    folders = [
        OUTPUT_DIR,
        CLEANED_DATA_DIR,
        CHARTS_DIR,
        POWERBI_DIR,
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def main():
    """
    Run full Avon River analytics workflow.
    """

    create_output_folders()

    # Step 1: load, clean and merge data
    cleaner = DataCleaner(
        input_file=INPUT_FILE,
        output_dir=CLEANED_DATA_DIR,
    )

    cleaner.load_raw_data()
    cleaner.split_datasets()
    cleaner.clean_water_quality_data()
    cleaner.clean_fish_population_data()
    cleaner.detect_outliers()
    cleaner.merge_datasets()
    cleaner.save_cleaned_datasets()

    # Step 2: run statistics and machine learning
    ml = MLAnalyzer(
        wq=cleaner.wq,
        fp=cleaner.fp,
        merged=cleaner.merged,
        wq_stats=cleaner.wq_stats,
        fp_stats=cleaner.fp_stats,
        cleaning_log=cleaner.cleaning_log,
        outlier_summary=cleaner.outlier_summary,
    )

    ml.run_correlation_analysis()
    ml.run_kmeans_clustering()
    ml.run_random_forest_regression()
    ml.run_trend_analysis()

    # Step 3: generate all charts for report
    charts = ChartGenerator(
        wq=cleaner.wq,
        fp=cleaner.fp,
        merged=cleaner.merged,
        wq_stats=cleaner.wq_stats,
        fp_stats=cleaner.fp_stats,
        importance=ml.importance,
        trend_results=ml.trend_results,
        r2=ml.r2,
        rmse=ml.rmse,
        output_dir=CHARTS_DIR,
    )

    charts.generate_all()

    # Step 4: export Power BI-ready CSV files
    dashboard = PowerBIExporter(
        cleaner=cleaner,
        ml=ml,
        output_dir=POWERBI_DIR,
    )

    dashboard.export_powerbi_files()

    # Step 5: print final summary
    print("\nPipeline completed successfully.")
    print(f"Cleaned data folder: {CLEANED_DATA_DIR}")
    print(f"Charts folder: {CHARTS_DIR}")
    print(f"Power BI folder: {POWERBI_DIR}")


if __name__ == "__main__":
    main()