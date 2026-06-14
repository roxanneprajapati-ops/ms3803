"""
This file creates report charts.
It receives cleaned data and ML results from main.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


COLORS = {
    "AV-1": "#2E75B6",
    "AV-2": "#70AD47",
    "AV-3": "#C00000",
}

CLUSTER_COLORS = {
    "Good": "#70AD47",
    "Moderate": "#ED7D31",
    "At-Risk": "#C00000",
}


class ChartGenerator:
    """
    ChartGenerator creates all charts for the report.
    This class only focus on visualisation. It does not clean data or train ML model.
    """

    def __init__(
        self,
        wq,
        fp,
        merged,
        wq_stats,
        fp_stats,
        importance,
        trend_results,
        r2,
        rmse,
        output_dir,
    ):
        """
        Store datasets and results needed for charts.
        """

        self.wq = wq
        self.fp = fp
        self.merged = merged
        self.wq_stats = wq_stats
        self.fp_stats = fp_stats
        self.importance = importance
        self.trend_results = trend_results
        self.r2 = r2
        self.rmse = rmse
        self.output_dir = output_dir
        self.chart_count = 0

        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_style()

    def setup_style(self):
        """
        Set chart style.
        This make all chart same look and feel.
        """

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#F9F9F9",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#E0E0E0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

    def print_step(self, title):
        """
        Print chart step title.
        """

        print("\n" + "*" * 35)
        print(title)
        print("*" * 35)

    def save_chart(self, file_name):
        """
        Save chart to output folder.
        """

        path = os.path.join(self.output_dir, file_name)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        self.chart_count += 1
        print(f"  [{self.chart_count}/10] Saved: {file_name}")

    def generate_all(self):
        """
        Generate all charts.
        Tables are printed only, not saved as images.
        """

        self.print_step("STEP 11: GENERATING CHARTS")

        self.generate_do_trend_chart()
        self.generate_fish_population_chart()
        self.generate_water_quality_chart()
        self.generate_correlation_chart()
        self.generate_do_vs_fish_scatter()
        self.generate_site_radar_chart()
        self.generate_monthly_average_chart()

        # this is table output only, not image
        self.print_risk_summary_table()

        self.generate_do_projection_chart()
        self.generate_ml_insights_chart()
        self.generate_actual_vs_predicted_chart()

        # this is table output only, not image
        self.print_project_comparison_table()

        self.print_chart_summary()

    def print_chart_summary(self):
        """
        Print final chart generation summary.
        """

        print("\nChart generation complete")
        print(f"Total chart images created: {self.chart_count}")
        print("Tables printed only: Risk Summary, Project A vs Project B")

    def generate_do_trend_chart(self):
        """
        Chart 1: dissolved oxygen trend.

        This shows DO readings and 7.0 mg/L danger line.
        """

        fig, ax = plt.subplots(figsize=(12, 6))

        self.plot_do_lines(ax)
        self.add_do_threshold(ax)
        self.format_do_trend_chart(fig, ax)

        self.save_chart("chart1_do_trend.png")

    def plot_do_lines(self, ax):
        """
        Plot DO line for each site.
        """

        for site in ["AV-1", "AV-2", "AV-3"]:
            data = self.wq[self.wq["Site"] == site].sort_values("Date")

            ax.plot(
                data["Date"],
                data["DO"],
                marker="o",
                markersize=5,
                linewidth=2,
                color=COLORS[site],
                label=f"Site {site}",
                alpha=0.85,
            )

    def add_do_threshold(self, ax):
        """
        Add 7.0 mg/L threshold line.
        """

        ax.axhline(
            y=7.0,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Native fish minimum (7.0 mg/L)",
        )

        ax.fill_between(
            self.wq["Date"].sort_values(),
            5,
            7.0,
            alpha=0.04,
            color="red",
        )

    def format_do_trend_chart(self, fig, ax):
        """
        Format DO trend chart labels.
        """

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=11)

        ax.set_title(
            "Dissolved Oxygen at AV-3 is Approaching Critical Levels\n"
            "Avon River Monitoring (October to December 2023)",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_ylim(5.5, 10.5)
        ax.legend(fontsize=9)
        fig.autofmt_xdate()
        plt.tight_layout()

    def generate_fish_population_chart(self):
        """
        Chart 2: fish population.
        This chart has fish count boxplot and species count by site.
        """

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))

        self.plot_fish_count_boxplot(axes[0])
        self.plot_species_bar_chart(axes[1])

        fig.suptitle(
            "AV-2 Supports the Healthiest Fish Community\n"
            "Avon River (October to December 2023)",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        self.save_chart("chart2_fish_count.png")

    def plot_fish_count_boxplot(self, ax):
        """
        Plot fish count distribution by site.
        """

        site_order = ["AV-1", "AV-2", "AV-3"]
        data = [self.fp[self.fp["Site"] == site]["Count"].values for site in site_order]

        box = ax.boxplot(
            data,
            labels=[f"Site {site}" for site in site_order],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )

        for patch, site in zip(box["boxes"], site_order):
            patch.set_facecolor(COLORS[site])
            patch.set_alpha(0.75)

        ax.set_ylabel("Fish Count per Observation", fontsize=10)
        ax.set_title("Fish Count Distribution by Site", fontsize=11, fontweight="bold")

    def plot_species_bar_chart(self, ax):
        """
        Plot species total count by site.
        """

        site_order = ["AV-1", "AV-2", "AV-3"]

        species_counts = (
            self.fp.groupby(["Site", "Species"])["Count"]
            .sum()
            .unstack(fill_value=0)
            .reindex(site_order)
        )

        palette = [
            "#2E75B6",
            "#70AD47",
            "#C00000",
            "#ED7D31",
            "#7030A0",
            "#888888",
            "#1F4E79",
        ]

        species_counts.plot(
            kind="bar",
            ax=ax,
            color=palette[:len(species_counts.columns)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlabel("Site", fontsize=10)
        ax.set_ylabel("Total Count", fontsize=10)
        ax.set_title("Total Fish Count by Species and Site", fontsize=11, fontweight="bold")
        ax.set_xticklabels([f"Site {site}" for site in site_order], rotation=0)
        ax.legend(fontsize=8, loc="upper right")

    def generate_water_quality_chart(self):
        """
        Chart 3: water quality dashboard.
        This shows temperature, pH and DO in one figure.
        """

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        params = [
            ("Temperature", "Temperature (°C)", (10, 23), None),
            ("pH", "pH", (6.8, 8.1), None),
            ("DO", "Dissolved Oxygen (mg/L)", (5.5, 10.5), 7.0),
        ]

        for index, item in enumerate(params):
            self.plot_water_quality_parameter(axes[index], item, index)

        fig.suptitle(
            "Water Quality Across All Sites. AV-3 Shows Highest Risk Profile\n"
            "Avon River Monitoring (October to December 2023)",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        self.save_chart("chart3_water_quality_dashboard.png")

    def plot_water_quality_parameter(self, ax, item, index):
        """
        Plot one water quality parameter.
        """

        col, label, ylim, threshold = item

        for site in ["AV-1", "AV-2", "AV-3"]:
            data = self.wq[self.wq["Site"] == site].sort_values("Date")

            ax.plot(
                data["Date"],
                data[col],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                color=COLORS[site],
                label=f"Site {site}",
                alpha=0.8,
            )

        if threshold:
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

            ax.fill_between(
                self.wq["Date"].sort_values(),
                ylim[0],
                threshold,
                alpha=0.04,
                color="red",
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(ylim)
        ax.tick_params(axis="x", rotation=30, labelsize=8)

        if index == 0:
            ax.legend(fontsize=8)

    def generate_correlation_chart(self):
        """
        Chart 4: correlation heatmap.
        """

        corr_df = self.merged[["Temperature", "pH", "DO", "Count", "AvgSize"]].copy()

        corr_df.columns = [
            "Temperature\n(°C)",
            "pH",
            "Dissolved\nOxygen",
            "Fish\nCount",
            "Avg Fish\nSize (cm)",
        ]

        fig, ax = plt.subplots(figsize=(9, 7))

        sns.heatmap(
            corr_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            center=0,
            linewidths=0.8,
            linecolor="white",
            annot_kws={"size": 12, "weight": "bold"},
            ax=ax,
            cbar_kws={"label": "Pearson Correlation Coefficient"},
        )

        ax.set_title(
            "Pearson Correlation Matrix\n"
            "Water Quality vs Fish Population(Avon River Clean Dataset)",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )

        plt.tight_layout()
        self.save_chart("chart4_correlation_matrix.png")

    def generate_do_vs_fish_scatter(self):
        """
        Chart 5: DO vs fish count scatter.

        This chart shows if fish count changes
        when dissolved oxygen changes.
        """

        fig, ax = plt.subplots(figsize=(10, 6))

        self.plot_do_fish_points(ax)
        self.add_do_fish_trend_line(ax)
        self.format_do_fish_scatter(ax)

        plt.tight_layout()
        self.save_chart("chart5_do_vs_fish_scatter.png")

    def plot_do_fish_points(self, ax):
        """
        Plot DO and fish count points by site.
        """

        for site in ["AV-1", "AV-2", "AV-3"]:
            data = self.merged[self.merged["Site"] == site]

            ax.scatter(
                data["DO"],
                data["Count"],
                color=COLORS[site],
                s=80,
                alpha=0.75,
                label=f"Site {site}",
                edgecolors="white",
                linewidth=0.8,
                zorder=4,
            )

    def add_do_fish_trend_line(self, ax):
        """
        Add linear trend line for DO and fish count.
        """

        slope, intercept, r_value, p_value, _ = stats.linregress(
            self.merged["DO"],
            self.merged["Count"],
        )

        x_line = np.linspace(
            self.merged["DO"].min() - 0.2,
            self.merged["DO"].max() + 0.2,
            100,
        )

        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="#333333",
            linewidth=1.8,
            linestyle="--",
            label=f"Linear trend (r={r_value:.2f}, p={p_value:.3f})",
            zorder=2,
        )

        ax.axvline(
            x=7.0,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.8,
            label="DO threshold (7.0 mg/L)",
        )

    def format_do_fish_scatter(self, ax):
        """
        Format DO vs fish scatter chart.
        """

        ax.set_xlabel("Dissolved Oxygen (mg/L)", fontsize=11)
        ax.set_ylabel("Fish Count per Observation", fontsize=11)

        ax.set_title(
            "Dissolved Oxygen vs Fish Count\n"
            "Scatter Plot with Linear Regression (Avon River 2023)",
            fontsize=13,
            fontweight="bold",
        )

        ax.legend(fontsize=9)

    def generate_site_radar_chart(self):
        """
        Chart 6: site radar chart.
        This compares water quality profile across the three sites.
        """

        site_means = self.wq.groupby("Site")[["Temperature", "pH", "DO"]].mean()
        norm = (site_means - site_means.min()) / (site_means.max() - site_means.min())

        categories = ["Temperature\n(°C)", "pH", "Dissolved\nOxygen (mg/L)"]
        angle_count = 3
        angles = [n / float(angle_count) * 2 * np.pi for n in range(angle_count)]
        angles += [0]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        for site in ["AV-1", "AV-2", "AV-3"]:
            values = norm.loc[site].tolist()
            values = values + [values[0]]

            ax.plot(
                angles,
                values,
                linewidth=2.5,
                color=COLORS[site],
                label=f"Site {site}",
            )
            ax.fill(angles, values, alpha=0.12, color=COLORS[site])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")

        ax.set_title(
            "Normalised Water Quality Profile by Site\n"
            "(Higher = relatively better condition)",
            fontsize=12,
            fontweight="bold",
            pad=24,
        )

        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)

        plt.tight_layout()
        self.save_chart("chart6_site_radar.png")

    def generate_monthly_average_chart(self):
        """
        Chart 7: monthly average chart.
        This shows monthly average changes for temperature, pH and DO.
        """

        self.wq["Month"] = self.wq["Date"].dt.to_period("M")

        monthly = (
            self.wq.groupby(["Site", "Month"])[["Temperature", "pH", "DO"]]
            .mean()
            .reset_index()
        )

        monthly["MonthStr"] = monthly["Month"].astype(str)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        chart_items = [
            ("Temperature", "Temperature (°C)", None),
            ("pH", "pH", None),
            ("DO", "Dissolved Oxygen (mg/L)", 7.0),
        ]

        for index, item in enumerate(chart_items):
            self.plot_monthly_parameter(axes[index], monthly, item, index)

        fig.suptitle(
            "Monthly Average Water Quality by Site\n"
            "Avon River (October to December 2023)",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        self.save_chart("chart7_parameter_change.png")

    def plot_monthly_parameter(self, ax, monthly, item, index):
        """
        Plot one monthly parameter.
        """

        col, label, threshold = item

        for site in ["AV-1", "AV-2", "AV-3"]:
            data = monthly[monthly["Site"] == site].sort_values("MonthStr")

            ax.plot(
                data["MonthStr"],
                data[col],
                marker="o",
                markersize=8,
                linewidth=2.2,
                color=COLORS[site],
                label=f"Site {site}",
            )

            # add value label so chart easy to read
            for _, row in data.iterrows():
                ax.annotate(
                    f"{row[col]:.1f}",
                    xy=(row["MonthStr"], row[col]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=COLORS[site],
                    fontweight="bold",
                )

        if threshold:
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month")

        if index == 0:
            ax.legend(fontsize=9)

    def risk_level(self, site):
        """
        Return risk level for site.
        This is based on DO minimum and mean.
        """

        do_min = self.wq_stats.loc[site, "DO_min"]
        do_mean = self.wq_stats.loc[site, "DO_mean"]

        if do_min < 6.5 or do_mean < 7.0:
            return "CRITICAL"
        if do_min < 7.0:
            return "HIGH"
        if do_mean < 7.5:
            return "MEDIUM"

        return "LOW"

    def action(self, site):
        """
        Return recommended action for site.
        """

        risk = self.risk_level(site)

        if risk == "CRITICAL":
            return "Immediate intervention required"
        if risk == "HIGH":
            return "Urgent targeted monitoring"
        if risk == "MEDIUM":
            return "Regular scheduled monitoring"

        return "Maintain current conditions"

    def print_risk_summary_table(self):
        """
        Print risk summary table only.
        No image/table chart will be created.
        """

        print("\nRisk Summary Table")
        print("-" * 100)

        table_data = self.build_risk_table_data()
        columns = self.get_risk_table_columns()

        print(" | ".join(columns))
        print("-" * 100)

        for row in table_data:
            print(" | ".join(str(value) for value in row))

    def build_risk_table_data(self):
        """
        Build table rows for risk summary chart.
        """

        table_data = []

        for site in ["AV-1", "AV-2", "AV-3"]:
            site_stats = self.wq_stats.loc[site]

            table_data.append([
                site,
                f"{site_stats['Temp_mean']:.1f} "
                f"({site_stats['Temp_min']:.1f} to {site_stats['Temp_max']:.1f})",
                f"{site_stats['pH_mean']:.2f} "
                f"(min {site_stats['pH_min']:.2f})",
                f"{site_stats['DO_mean']:.2f} "
                f"(min {site_stats['DO_min']:.2f})",
                str(int(self.fp_stats.loc[site, "species_n"])),
                f"{self.fp_stats.loc[site, 'count_mean']:.1f}",
                self.risk_level(site),
                self.action(site),
            ])

        return table_data

    def get_risk_table_columns(self):
        """
        Return risk table column names.
        """

        return [
            "Site",
            "Temp (°C) mean (range)",
            "pH mean (min)",
            "DO (mg/L) mean (min)",
            "Species Count",
            "Avg Fish Count",
            "Risk Level",
            "Recommended Action",
        ]

    def generate_do_projection_chart(self):
        """
        Chart 8: DO trend and 30-day projection.
        """

        fig, ax = plt.subplots(figsize=(12, 6))

        for site in ["AV-1", "AV-2", "AV-3"]:
            self.plot_site_projection(ax, site)

        self.add_projection_thresholds(ax)
        self.format_projection_chart(fig, ax)

        self.save_chart("chart8_do_projection.png")

    def plot_site_projection(self, ax, site):
        """
        Plot observed trend and projected line for one site.
        """

        data = self.wq[self.wq["Site"] == site].sort_values("Date").copy()

        data["days"] = (
            data["Date"] - data["Date"].min()
        ).dt.days

        slope, intercept, r_value, _, _ = stats.linregress(
            data["days"],
            data["DO"],
        )

        x_obs = np.linspace(0, data["days"].max(), 100)

        x_dates = [
            data["Date"].min() + pd.Timedelta(days=float(day))
            for day in x_obs
        ]

        ax.scatter(
            data["Date"],
            data["DO"],
            color=COLORS[site],
            s=25,
            alpha=0.5,
            zorder=4,
        )

        ax.plot(
            x_dates,
            slope * x_obs + intercept,
            "-",
            color=COLORS[site],
            linewidth=2.2,
            label=f"Site {site} trend (r={r_value:.2f})",
            zorder=3,
        )

        self.plot_projection_line(ax, data, slope, intercept, site)

    def plot_projection_line(self, ax, data, slope, intercept, site):
        """
        Plot 30-day projected DO line.
        """

        x_proj = np.linspace(
            data["days"].max(),
            data["days"].max() + 30,
            30,
        )

        x_proj_dates = [
            data["Date"].min() + pd.Timedelta(days=float(day))
            for day in x_proj
        ]

        y_proj = slope * x_proj + intercept

        ax.plot(
            x_proj_dates,
            y_proj,
            "--",
            color=COLORS[site],
            linewidth=1.5,
            alpha=0.55,
        )

        ax.fill_between(
            x_proj_dates,
            y_proj - 0.3,
            y_proj + 0.3,
            alpha=0.08,
            color=COLORS[site],
        )

    def add_projection_thresholds(self, ax):
        """
        Add DO threshold lines in projection chart.
        """

        ax.axhline(
            y=7.0,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label="Min. threshold (7.0 mg/L)",
        )

        ax.axhline(
            y=5.0,
            color="#8B0000",
            linestyle=":",
            linewidth=1.2,
            alpha=0.7,
            label="Hypoxia threshold (5.0 mg/L)",
        )

    def format_projection_chart(self, fig, ax):
        """
        Format projection chart.
        """

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Dissolved Oxygen (mg/L)", fontsize=11)

        ax.set_title(
            "DO Trend and 30-Day Projection. AV-3 Requires Urgent Action\n"
            "(Solid = observed trend | Dashed = 30-day projection)",
            fontsize=13,
            fontweight="bold",
        )

        ax.set_ylim(4.0, 11.5)
        ax.legend(fontsize=9)
        fig.autofmt_xdate()

        fig.text(
            0.5,
            -0.02,
            "Note: Projections assume a linear trend. "
            "Seasonal variation and interventions may alter this trajectory.",
            ha="center",
            fontsize=8.5,
            color="#555555",
            style="italic",
        )

        plt.tight_layout()

    def generate_ml_insights_chart(self):
        """
        Chart 9: ML insights.
        This combines K-Means cluster chart and Random Forest feature importance.
        """

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        self.plot_cluster_scatter(axes[0])
        self.plot_feature_importance(axes[1])

        fig.suptitle(
            "Machine Learning Insights\n"
            "Avon River Dataset (October to December 2023)",
            fontsize=13,
            fontweight="bold",
        )

        plt.tight_layout()
        self.save_chart("chart9_ml_insights.png")

    def plot_cluster_scatter(self, ax):
        """
        Plot K-Means health zones.
        """

        for cluster in ["Good", "Moderate", "At-Risk"]:
            data = self.merged[self.merged["Cluster"] == cluster]

            ax.scatter(
                data["DO"],
                data["Temperature"],
                color=CLUSTER_COLORS[cluster],
                s=90,
                alpha=0.8,
                label=cluster,
                edgecolors="white",
                linewidth=0.8,
                zorder=4,
            )

        ax.axvline(
            x=7.0,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.set_xlabel("Dissolved Oxygen (mg/L)", fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=11)

        ax.set_title(
            "K-Means: Three Ecological Health Zones\n"
            "DO and Temperature as primary separators",
            fontsize=11,
            fontweight="bold",
        )

        ax.legend(title="Health Zone", fontsize=9)

    def plot_feature_importance(self, ax):
        """
        Plot Random Forest feature importance.
        """

        imp_sorted = self.importance.sort_values("Importance", ascending=True)

        colors = [
            "#2E75B6" if feature == "DO"
            else "#70AD47" if feature == "Temperature"
            else "#C00000"
            for feature in imp_sorted["Feature"]
        ]

        bars = ax.barh(
            imp_sorted["Feature"],
            imp_sorted["Importance"] * 100,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.2,
        )

        for bar, value in zip(bars, imp_sorted["Importance"] * 100):
            ax.text(
                value + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#333333",
            )

        ax.set_xlabel("Feature Importance (%)", fontsize=11)
        ax.set_title(
            "Random Forest: What Drives Fish Population Health?\n"
            "All three parameters matter — manage them together",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlim(0, 55)

    def generate_actual_vs_predicted_chart(self):
        """
        Chart 10: actual vs predicted fish count.
        """

        fig, ax = plt.subplots(figsize=(8, 6))

        for site in ["AV-1", "AV-2", "AV-3"]:
            data = self.merged[self.merged["Site"] == site]

            ax.scatter(
                data["Count"],
                data["Predicted_Count"],
                color=COLORS[site],
                s=80,
                alpha=0.75,
                label=f"Site {site}",
                edgecolors="white",
                linewidth=0.8,
            )

        max_value = max(
            self.merged["Count"].max(),
            self.merged["Predicted_Count"].max(),
        ) + 5

        ax.plot(
            [0, max_value],
            [0, max_value],
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label="Perfect prediction",
        )

        ax.set_xlabel("Actual Fish Count", fontsize=11)
        ax.set_ylabel("Predicted Fish Count", fontsize=11)

        ax.set_title(
            f"Random Forest Actual vs Predicted\n"
            f"Train R² = {self.r2:.3f} | RMSE = {self.rmse:.1f} fish",
            fontsize=12,
            fontweight="bold",
        )

        ax.legend(fontsize=9)
        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)

        plt.tight_layout()
        self.save_chart("chart10_actual_vs_predicted.png")

    def print_project_comparison_table(self):
        """
        Print Project A vs Project B comparison only.
        No image/table chart will be created.
        """

        print("\nProject A vs Project B Comparison")
        print("-" * 100)

        categories = [
            "Data Type",
            "Data Source",
            "Volume Required",
            "Data Recency",
            "Primary Risk",
            "Time to Insight",
            "Analytical Approach",
        ]

        project_a = [
            "Historical CRM, purchase, behavioural",
            "Internal CRM, market research",
            "Very High (ML training)",
            "Medium (historical patterns)",
            "Algorithmic bias, overfitting",
            "Long (months)",
            "Predictive modelling, ML / AI",
        ]

        project_b = [
            "Real-time behavioural, qualitative feedback",
            "A/B tests, user interviews, NPS",
            "Moderate (per experiment)",
            "Very High (current UX)",
            "Local optimisation, survivorship bias",
            "Short (weeks)",
            "Experimental design, statistical testing",
        ]

        print("Category | Project A | Project B")
        print("-" * 100)

        for category, a_value, b_value in zip(categories, project_a, project_b):
            print(f"{category} | {a_value} | {b_value}")