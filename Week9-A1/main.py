from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Fitness_App_User_Data.xlsx"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values, duplicates, data types, and basic inconsistencies."""
    clean = df.copy()

    # Standardise column names
    clean.columns = clean.columns.str.strip().str.replace(" ", "_")

    # Remove duplicate rows
    before_duplicates = len(clean)
    clean = clean.drop_duplicates()
    removed_duplicates = before_duplicates - len(clean)

    # Correct data types
    numeric_cols = [
        "Age",
        "Workouts_per_Week",
        "Avg_Session_Duration_Min",
        "Steps_per_Day",
        "Churned",
    ]
    categorical_cols = ["Gender", "Subscription_Type"]

    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

    for col in categorical_cols:
        if col in clean.columns:
            clean[col] = clean[col].astype(str).str.strip().str.title()
            clean[col] = clean[col].replace({"Nan": np.nan, "None": np.nan, "": np.nan})

    # Handle missing values
    for col in numeric_cols:
        if col in clean.columns:
            clean[col] = clean[col].fillna(clean[col].median())

    for col in categorical_cols:
        if col in clean.columns:
            mode_value = clean[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            clean[col] = clean[col].fillna(fill_value)

    # Remove impossible/invalid values where appropriate
    if "Age" in clean.columns:
        clean = clean[(clean["Age"] >= 13) & (clean["Age"] <= 100)]
    if "Workouts_per_Week" in clean.columns:
        clean = clean[(clean["Workouts_per_Week"] >= 0) & (clean["Workouts_per_Week"] <= 14)]
    if "Avg_Session_Duration_Min" in clean.columns:
        clean = clean[(clean["Avg_Session_Duration_Min"] >= 0) & (clean["Avg_Session_Duration_Min"] <= 300)]
    if "Steps_per_Day" in clean.columns:
        clean = clean[(clean["Steps_per_Day"] >= 0) & (clean["Steps_per_Day"] <= 100000)]

    print("DATA CLEANING SUMMARY")
    print(f"Original rows: {len(df)}")
    print(f"Rows after cleaning: {len(clean)}")
    print(f"Duplicates removed: {removed_duplicates}")
    print("Missing values after cleaning:")
    print(clean.isna().sum())
    print()

    return clean.reset_index(drop=True)


def build_preprocessor():
    numeric_features = [
        "Age",
        "Workouts_per_Week",
        "Avg_Session_Duration_Min",
        "Steps_per_Day",
    ]
    categorical_features = ["Gender", "Subscription_Type"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def choose_optimal_k(x_processed, k_range=range(2, 9)):
    inertias = []
    silhouettes = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_processed)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(x_processed, labels))

    results = pd.DataFrame({"k": list(k_range), "inertia": inertias, "silhouette": silhouettes})
    best_k = int(results.loc[results["silhouette"].idxmax(), "k"])

    print("CLUSTER SELECTION")
    print(results.to_string(index=False))
    print(f"Selected k based on highest silhouette score: {best_k}")
    print()

    return best_k, results


def save_cluster_plots(x_processed, labels, k_results, clustered_df):
    # Elbow chart
    plt.figure(figsize=(8, 5))
    plt.plot(k_results["k"], k_results["inertia"], marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow.png", dpi=200)
    plt.close()

    # Silhouette chart
    plt.figure(figsize=(8, 5))
    plt.plot(k_results["k"], k_results["silhouette"], marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette.png", dpi=200)
    plt.close()

    # PCA cluster visualisation
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(x_processed)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, alpha=0.75)
    plt.title("User Clusters Visualised with PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_clusters.png", dpi=200)
    plt.close()

    # Cluster profile chart
    profile_cols = ["Age", "Workouts_per_Week", "Avg_Session_Duration_Min", "Steps_per_Day"]
    profile = clustered_df.groupby("Cluster")[profile_cols].mean()
    profile_scaled = (profile - profile.mean()) / profile.std(ddof=0)

    plt.figure(figsize=(9, 5))
    profile_scaled.T.plot(kind="bar", ax=plt.gca())
    plt.title("Cluster Profiles using Standardised Feature Averages")
    plt.xlabel("Feature")
    plt.ylabel("Relative Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "profile.png", dpi=200)
    plt.close()

    # Churn rate by cluster
    if "Churned" in clustered_df.columns:
        churn_rate = clustered_df.groupby("Cluster")["Churned"].mean().sort_index() * 100
        plt.figure(figsize=(7, 5))
        churn_rate.plot(kind="bar")
        plt.title("Churn Rate by Cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "churn.png", dpi=200)
        plt.close()


def main():
    df = load_data(INPUT_FILE)
    clean = clean_data(df)

    selected_features = [
        "Age",
        "Gender",
        "Workouts_per_Week",
        "Avg_Session_Duration_Min",
        "Steps_per_Day",
        "Subscription_Type",
    ]

    x = clean[selected_features]
    preprocessor = build_preprocessor()
    x_processed = preprocessor.fit_transform(x)

    best_k, k_results = choose_optimal_k(x_processed)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clean["Cluster"] = kmeans.fit_predict(x_processed)

    summary = clean.groupby("Cluster").agg(
        Users=("User_ID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Workouts_per_Week=("Workouts_per_Week", "mean"),
        Avg_Session_Duration_Min=("Avg_Session_Duration_Min", "mean"),
        Avg_Steps_per_Day=("Steps_per_Day", "mean"),
        Churn_Rate=("Churned", "mean"),
    )
    summary["Churn_Rate"] = summary["Churn_Rate"] * 100
    summary = summary.round(2)

    print("CLUSTER SUMMARY")
    print(summary.to_string())
    print()

    clean.to_csv(BASE_DIR / "Fitness_App_User_Data_Cleaned_Clustered.csv", index=False)
    summary.to_csv(BASE_DIR / "Fitness_App_Cluster_Summary.csv")

    save_cluster_plots(x_processed, clean["Cluster"].values, k_results, clean)

    print("Files saved:")
    print(BASE_DIR / "Fitness_App_User_Data_Cleaned_Clustered.csv")
    print(BASE_DIR / "Fitness_App_Cluster_Summary.csv")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
