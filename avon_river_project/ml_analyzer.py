"""
Machine learning and statistical analysis class.

This file handles:
- Pearson correlation
- K-Means clustering
- Random Forest regression
- trend analysis
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import pandas as pd
import numpy as np

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


class MLAnalyzer:
    """
    MLAnalyzer runs statistics and machine learning.
    It receives cleaned data from DataCleaner.
    """

    def __init__(
        self,
        wq,
        fp,
        merged,
        wq_stats,
        fp_stats,
        cleaning_log,
        outlier_summary,
    ):
        self.wq = wq
        self.fp = fp
        self.merged = merged
        self.wq_stats = wq_stats
        self.fp_stats = fp_stats

        self.cleaning_log = cleaning_log
        self.outlier_summary = outlier_summary

        self.corr_matrix = None
        self.inertias = []
        self.importance = None
        self.trend_results = {}

        self.rf = None
        self.x_rf = None
        self.y_rf = None

        self.r2 = None
        self.rmse = None
        self.loo_mean = None
        self.loo_std = None

        self.r_do = None
        self.p_do = None
        self.r_temp = None
        self.p_temp = None
        self.r_ph = None
        self.p_ph = None

    def print_step(self, title):
        print("\n" + "*" * 35)
        print(title)
        print("-" * 35)

    def run_correlation_analysis(self):
        self.print_step("STEP 7: CORRELATION ANALYSIS")
        self.calculate_correlation_matrix()
        self.calculate_key_correlations()
        self.print_correlation_summary()

    def calculate_correlation_matrix(self):
        corr_cols = ["Temperature", "pH", "DO", "Count", "AvgSize"]
        self.corr_matrix = self.merged[corr_cols].corr().round(3)

        print("\n  Pearson Correlation Matrix:")
        print(self.corr_matrix.to_string())

    def calculate_key_correlations(self):
        self.r_do, self.p_do = stats.pearsonr(
            self.merged["DO"],
            self.merged["Count"],
        )

        self.r_temp, self.p_temp = stats.pearsonr(
            self.merged["Temperature"],
            self.merged["Count"],
        )

        self.r_ph, self.p_ph = stats.pearsonr(
            self.merged["pH"],
            self.merged["Count"],
        )

    def print_correlation_summary(self):
        print(
            f"\n  DO   vs Count: r={self.r_do:.3f}, "
            f"p={self.p_do:.4f}"
        )

        print(
            f"  Temp vs Count: r={self.r_temp:.3f}, "
            f"p={self.p_temp:.4f}"
        )

        print(
            f"  pH   vs Count: r={self.r_ph:.3f}, "
            f"p={self.p_ph:.4f}"
        )

        print("\n  Note: weak correlation means ML is useful")
        print("  because many factors may work together.")

    def run_kmeans_clustering(self):
        self.print_step("STEP 8: K-MEANS CLUSTERING")
        x_scaled = self.scale_cluster_features()
        self.calculate_elbow_curve(x_scaled)
        self.fit_final_clusters(x_scaled)
        self.print_cluster_summary()

    def scale_cluster_features(self):
        cluster_features = ["Temperature", "pH", "DO", "Count", "AvgSize"]
        x_cluster = self.merged[cluster_features].copy()

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_cluster)

        return x_scaled

    def calculate_elbow_curve(self, x_scaled):
        print("\n  Elbow method:")

        self.inertias = []

        for k in range(2, 8):
            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
            )

            km.fit(x_scaled)

            self.inertias.append({
                "k": k,
                "inertia": round(km.inertia_, 2),
            })

            print(f"    k={k}: inertia={km.inertia_:.2f}")

        print("  k=3 selected for Good, Moderate, At-Risk groups")

    def fit_final_clusters(self, x_scaled):
        km3 = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10,
        )

        self.merged["Cluster_raw"] = km3.fit_predict(x_scaled)
        self.label_clusters_by_do()

    def label_clusters_by_do(self):
        cluster_do = (
            self.merged.groupby("Cluster_raw")["DO"]
            .mean()
            .sort_values(ascending=False)
        )

        label_map = {}

        for i, (cluster_id, _) in enumerate(cluster_do.items()):
            label_map[cluster_id] = ["Good", "Moderate", "At-Risk"][i]

        self.merged["Cluster"] = self.merged["Cluster_raw"].map(label_map)

    def print_cluster_summary(self):
        cluster_profile = self.merged.groupby("Cluster").agg(
            n=("DO", "count"),
            DO_mean=("DO", "mean"),
            DO_min=("DO", "min"),
            Temp_mean=("Temperature", "mean"),
            pH_mean=("pH", "mean"),
            Count_mean=("Count", "mean"),
            Size_mean=("AvgSize", "mean"),
        ).round(2)

        site_cluster = pd.crosstab(
            self.merged["Site"],
            self.merged["Cluster"],
        )

        print("\n  Cluster Profiles:")
        print(cluster_profile.to_string())

        print("\n  Site-Cluster Distribution:")
        print(site_cluster.to_string())

    def run_random_forest_regression(self):
        self.print_step("STEP 9: RANDOM FOREST REGRESSION")

        self.prepare_rf_data()
        self.train_rf_model()
        self.calculate_feature_importance()
        self.skip_loo_validation()
        self.calculate_predictions()
        self.print_rf_summary()

    def prepare_rf_data(self):
        rf_features = ["Temperature", "pH", "DO"]

        self.x_rf = self.merged[rf_features].values
        self.y_rf = self.merged["Count"].values

    def train_rf_model(self):
        self.rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        )

        self.rf.fit(
            self.x_rf,
            self.y_rf,
        )

    def calculate_feature_importance(self):
        features = ["Temperature", "pH", "DO"]

        self.importance = pd.DataFrame({
            "Feature": features,
            "Importance": self.rf.feature_importances_.round(4),
        }).sort_values("Importance", ascending=False)

    def skip_loo_validation(self):
        """
        LOOCV R² is skipped because R² is not valid
        for single-observation test folds.
        """

        self.loo_mean = None
        self.loo_std = None

        print(
            "\n  LOOCV R² skipped "
            "(not valid for single-observation test folds)"
        )

    def calculate_predictions(self):
        y_pred = self.rf.predict(self.x_rf)

        self.r2 = r2_score(self.y_rf, y_pred)
        self.rmse = np.sqrt(
            mean_squared_error(
                self.y_rf,
                y_pred
            )
        )

        self.merged["Predicted_Count"] = y_pred.round(1)

    def print_rf_summary(self):
        print("\n  Feature Importance:")

        for _, row in self.importance.iterrows():
            bar = "█" * int(row["Importance"] * 50)
            print(
                f"  {row['Feature']:<12} "
                f"{row['Importance'] * 100:5.1f}%  {bar}"
            )

        print("\n  LOO CV R²: Skipped")
        print("  Reason: R² is not valid with one test sample per fold")
        print(f"  Train R²: {self.r2:.3f}")
        print(f"  Train RMSE: {self.rmse:.2f} fish")

    def run_trend_analysis(self):
        self.print_step("STEP 10: TREND ANALYSIS")

        self.trend_results = {}

        for site in ["AV-1", "AV-2", "AV-3"]:
            self.analyse_site_trend(site)

    def analyse_site_trend(self, site):
        df_site = self.wq[
            self.wq["Site"] == site
        ].sort_values("Date").copy()

        df_site["days"] = (
            df_site["Date"] - df_site["Date"].min()
        ).dt.days

        slope, intercept, r_value, p_value, _ = stats.linregress(
            df_site["days"],
            df_site["DO"],
        )

        direction = self.get_trend_direction(slope)

        self.trend_results[site] = {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
            "r": round(r_value, 3),
            "p": round(p_value, 4),
            "direction": direction,
            "start_date": df_site["Date"].min().strftime("%Y-%m-%d"),
            "days": int(df_site["days"].max()),
        }

        print(
            f"  {site}: slope={slope:+.4f} mg/L/day | "
            f"r={r_value:.3f} | p={p_value:.4f} | {direction}"
        )

    def get_trend_direction(self, slope):
        if slope < -0.001:
            return "Declining"

        if abs(slope) < 0.001:
            return "Stable"

        return "Improving"