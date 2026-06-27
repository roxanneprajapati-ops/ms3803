"""
Week 9 - Activity 2: User Knowledge Modeling Dataset
"""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# Dataset columns used in the analysis
FEATURES = ["STG", "SCG", "STR", "LPR", "PEG"]
TARGET = "UNS"


def get_paths() -> dict:
    """Create all project paths using the location of this Python file."""
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "output"
    charts_dir = output_dir / "charts"
    tables_dir = output_dir / "tables"

    charts_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_dir,
        "data": base_dir / "Data_User_Modeling_Dataset_Hamdi Tolga KAHRAMAN.xls",
        "output": output_dir,
        "charts": charts_dir,
        "tables": tables_dir,
    }


def load_and_clean_sheet(file_path: Path, sheet_name: str) -> pd.DataFrame:
    """Load one Excel sheet and keep only the useful columns."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Fix spaces in column names such as " UNS"
    df.columns = [str(col).strip() for col in df.columns]

    # Remove note columns and keep only modelling columns
    df = df[FEATURES + [TARGET]].copy()

    # Make target labels consistent: "Very Low" -> "very_low"
    df[TARGET] = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Convert feature values to numeric and remove invalid rows
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_and_explore_data(paths: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training and test data, then save a basic data summary."""
    train_df = load_and_clean_sheet(paths["data"], "Training_Data")
    test_df = load_and_clean_sheet(paths["data"], "Test_Data")
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    summary = {
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "total_rows": len(full_df),
        "features": FEATURES,
        "target": TARGET,
        "class_counts_all_data": full_df[TARGET].value_counts().to_dict(),
        "missing_values_all_data": full_df.isna().sum().to_dict(),
        "duplicate_rows_all_data": int(full_df.duplicated().sum()),
    }

    with open(paths["output"] / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    full_df.describe().round(3).to_csv(paths["tables"] / "feature_summary.csv")
    full_df.to_csv(paths["tables"] / "processed_user_modeling_data.csv", index=False)
    return train_df, test_df, full_df


def train_and_evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame, paths: dict) -> tuple[str, LabelEncoder, np.ndarray]:
    """Train classification models and save evaluation results."""
    x_train = train_df[FEATURES]
    x_test = test_df[FEATURES]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET])
    y_test = label_encoder.transform(test_df[TARGET])

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "SVM (RBF)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
            ]
        ),
        "KNN (k=5)": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }

    rows = []
    predictions = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        predictions[name] = y_pred

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "CV Accuracy Mean": cv_scores.mean(),
                "CV Accuracy Std": cv_scores.std(),
            }
        )

    performance_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    performance_df.round(4).to_csv(paths["tables"] / "model_performance.csv", index=False)

    best_model_name = performance_df.iloc[0]["Model"]
    best_predictions = predictions[best_model_name]
    class_names = list(label_encoder.classes_)

    # Save confusion matrix and detailed classification report
    cm = confusion_matrix(y_test, best_predictions, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(paths["tables"] / "confusion_matrix.csv")

    report_df = pd.DataFrame(
        classification_report(
            y_test,
            best_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.round(4).to_csv(paths["tables"] / "classification_report.csv")

    # Save feature importance from Random Forest
    rf_model = models["Random Forest"]
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance_df.round(4).to_csv(paths["tables"] / "feature_importance.csv", index=False)

    return best_model_name, label_encoder, best_predictions


def run_clustering(full_df: pd.DataFrame, label_encoder: LabelEncoder, paths: dict) -> pd.DataFrame:
    """Run K-Means clustering and compare clusters with known knowledge labels."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(full_df[FEATURES])
    true_labels = label_encoder.transform(full_df[TARGET])

    results = []
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(x_scaled)
        results.append(
            {
                "k": k,
                "Inertia": kmeans.inertia_,
                "Silhouette": silhouette_score(x_scaled, cluster_labels),
                "ARI_vs_UNS": adjusted_rand_score(true_labels, cluster_labels),
            }
        )

    cluster_eval_df = pd.DataFrame(results)
    cluster_eval_df.round(4).to_csv(paths["tables"] / "clustering_evaluation.csv", index=False)

    # Four clusters are used because the target has four knowledge levels
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    full_df = full_df.copy()
    full_df["Cluster"] = kmeans.fit_predict(x_scaled)

    profile_df = pd.crosstab(full_df["Cluster"], full_df[TARGET], normalize="index").round(3)
    profile_df.to_csv(paths["tables"] / "cluster_profile_by_class.csv")

    pca = PCA(n_components=2, random_state=42)
    pca_values = pca.fit_transform(x_scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": pca_values[:, 0],
            "PC2": pca_values[:, 1],
            TARGET: full_df[TARGET],
            "Cluster": full_df["Cluster"],
        }
    )
    pca_df.to_csv(paths["tables"] / "pca_clusters.csv", index=False)

    full_df.to_csv(paths["tables"] / "processed_user_modeling_data.csv", index=False)
    return full_df


def generate_charts(full_df: pd.DataFrame, paths: dict, best_model_name: str) -> None:
    """Create charts for the report and presentation."""
    charts_dir = paths["charts"]
    tables_dir = paths["tables"]

    # Class distribution chart
    plt.figure(figsize=(7, 4.2))
    class_order = ["very_low", "low", "middle", "high"]
    full_df[TARGET].value_counts().reindex(class_order).plot(kind="bar")
    plt.title("Knowledge Level Distribution")
    plt.xlabel("Knowledge level")
    plt.ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(charts_dir / "class_distribution.png", dpi=200)
    plt.close()

    # Classification performance chart
    performance_df = pd.read_csv(tables_dir / "model_performance.csv")
    plt.figure(figsize=(7.5, 4.5))
    performance_df.set_index("Model")[["Accuracy", "F1 Score"]].plot(kind="bar", ax=plt.gca())
    plt.title("Classification Model Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(charts_dir / "model_performance.png", dpi=200)
    plt.close()

    # Confusion matrix chart
    cm_df = pd.read_csv(tables_dir / "confusion_matrix.csv", index_col=0)
    cm = cm_df.values
    labels = list(cm_df.index)
    plt.figure(figsize=(6, 5))
    image = plt.imshow(cm, cmap="Blues")
    plt.colorbar(image, fraction=0.046)
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(charts_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    # Correlation heatmap
    corr = full_df[FEATURES].corr()
    plt.figure(figsize=(6, 5))
    image = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(image, fraction=0.046)
    plt.title("Feature Correlation Heatmap")
    plt.xticks(range(len(FEATURES)), FEATURES)
    plt.yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    # Feature importance chart
    importance_df = pd.read_csv(tables_dir / "feature_importance.csv")
    importance_df = importance_df.sort_values("Importance", ascending=True)
    plt.figure(figsize=(7, 4.2))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(charts_dir / "feature_importance.png", dpi=200)
    plt.close()

    # K-Means PCA scatter plot
    pca_df = pd.read_csv(tables_dir / "pca_clusters.csv")
    plt.figure(figsize=(6.5, 5))
    scatter = plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["Cluster"], s=35, alpha=0.85)
    plt.title("K-Means Clusters (PCA View)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(charts_dir / "kmeans_clusters_pca.png", dpi=200)
    plt.close()

    # Silhouette chart
    cluster_eval_df = pd.read_csv(tables_dir / "clustering_evaluation.csv")
    plt.figure(figsize=(7, 4.2))
    plt.plot(cluster_eval_df["k"], cluster_eval_df["Silhouette"], marker="o")
    plt.title("K-Means Silhouette by k")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.xticks(range(2, 7))
    plt.tight_layout()
    plt.savefig(charts_dir / "cluster_silhouette.png", dpi=200)
    plt.close()


def save_final_summary(paths: dict, best_model_name: str) -> None:
    """Save a short summary of the most important results."""
    performance_df = pd.read_csv(paths["tables"] / "model_performance.csv")
    cluster_eval_df = pd.read_csv(paths["tables"] / "clustering_evaluation.csv")

    summary = {
        "best_classification_model": best_model_name,
        "best_accuracy": float(performance_df.iloc[0]["Accuracy"]),
        "best_f1_score": float(performance_df.iloc[0]["F1 Score"]),
        "best_k_by_silhouette": int(cluster_eval_df.loc[cluster_eval_df["Silhouette"].idxmax(), "k"]),
        "best_silhouette_score": float(cluster_eval_df["Silhouette"].max()),
    }

    with open(paths["output"] / "final_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    """Run the complete data analysis workflow."""
    paths = get_paths()

    if not paths["data"].exists():
        raise FileNotFoundError(
            f"Dataset not found: {paths['data']}\n"
            "Place the Excel file in the same folder as this Python script."
        )

    print("Step 1: Load and Explore Data")
    train_df, test_df, full_df = load_and_explore_data(paths)

    print("Step 2: Preprocess and Clean Data")
    print("Cleaned training rows:", len(train_df))
    print("Cleaned test rows:", len(test_df))

    print("Step 3: Classification Model Training and Evaluation")
    best_model_name, label_encoder, _ = train_and_evaluate_models(train_df, test_df, paths)

    print("Step 4: Clustering Analysis")
    clustered_df = run_clustering(full_df, label_encoder, paths)

    print("Step 5: Generate Charts and Save Results")
    generate_charts(clustered_df, paths, best_model_name)
    save_final_summary(paths, best_model_name)

    print("Analysis complete.")
    print(f"Best model: {best_model_name}")
    print(f"Outputs saved in: {paths['output']}")


if __name__ == "__main__":
    main()
