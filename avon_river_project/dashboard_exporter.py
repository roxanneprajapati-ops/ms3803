"""
Power BI exporter for Avon River assessment.
This file exports only the CSV files needed for the Power BI dashboard.
"""

import os
import pandas as pd


class PowerBIExporter:
    """
    Export Power BI-ready data.
    This class receives DataCleaner and MLAnalyzer objects.
    It uses their cleaned data and analysis results.
    """

    def __init__(self, cleaner, ml, output_dir):
        self.cleaner = cleaner
        self.ml = ml
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def print_step(self, title):
        print("\n" + "*" * 35)
        print(title)
        print("*" * 35)

    def export_powerbi_files(self):
        """
        Export only the CSV files required for the Power BI dashboard.
        """

        self.print_step("STEP 12: EXPORTING POWER BI FILES")

        # Core datasets used by Power BI visuals
        self.save_csv(
            self.cleaner.wq,
            "water_quality_clean.csv"
        )

        self.save_csv(
            self.cleaner.merged,
            "avon_river_merged_clean.csv"
        )

        # Summary/dashboard datasets
        self.save_csv(
            pd.DataFrame([self.build_kpi_data()]),
            "kpi_summary.csv"
        )

        self.save_csv(
            pd.DataFrame(self.build_risk_table_data()),
            "site_risk_summary.csv"
        )

        self.save_csv(
            pd.DataFrame(self.build_feature_importance_data()),
            "feature_importance.csv"
        )

        self.save_csv(
            pd.DataFrame(self.build_do_projection_data()),
            "do_projection.csv"
        )

        print("  Power BI CSV export completed.")
        print(f"  Folder: {self.output_dir}")

    def save_csv(self, df, filename):
        """
        Save dataframe to CSV.
        """

        data = df.copy()

        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = data[col].dt.strftime("%Y-%m-%d")

        path = os.path.join(
            self.output_dir,
            filename
        )

        data.to_csv(
            path,
            index=False
        )

        print(f"  Saved: {filename}")

    def build_kpi_data(self):
        """
        Build KPI values for Power BI cards.
        """

        wq = self.cleaner.wq
        fp = self.cleaner.fp

        return {
            "avg_do_av3": round(
                float(wq[wq["Site"] == "AV-3"]["DO"].mean()),
                2
            ),
            "min_do_av3": round(
                float(wq[wq["Site"] == "AV-3"]["DO"].min()),
                2
            ),
            "sites_at_risk": self.count_sites_at_risk(),
            "total_species": int(fp["Species"].nunique()),
            "total_fish": int(fp["Count"].sum()),
            "date_range_start": wq["Date"].min().strftime("%Y-%m-%d"),
            "date_range_end": wq["Date"].max().strftime("%Y-%m-%d"),
            "total_obs": int(len(wq)),
        }

    def build_feature_importance_data(self):
        """
        Build Random Forest feature importance data.
        """

        return self.ml.importance.to_dict("records")

    def build_risk_table_data(self):
        """
        Build site risk summary for Power BI.
        """

        result = []

        for site in ["AV-1", "AV-2", "AV-3"]:
            wq_stats = self.cleaner.wq_stats.loc[site]
            fp_stats = self.cleaner.fp_stats.loc[site]
            risk = self.risk_level(site)

            result.append({
                "site": site,
                "temp_mean": float(wq_stats["Temp_mean"]),
                "temp_min": float(wq_stats["Temp_min"]),
                "temp_max": float(wq_stats["Temp_max"]),
                "ph_mean": float(wq_stats["pH_mean"]),
                "ph_min": float(wq_stats["pH_min"]),
                "do_mean": float(wq_stats["DO_mean"]),
                "do_min": float(wq_stats["DO_min"]),
                "do_max": float(wq_stats["DO_max"]),
                "species_count": int(fp_stats["species_n"]),
                "avg_fish_count": float(fp_stats["count_mean"]),
                "risk": risk,
                "status": self.status_label(risk),
                "action": self.action(site),
            })

        return result

    def build_do_projection_data(self):
        """
        Build observed trend and 30-day DO projection data for Power BI.
        """

        result = []

        for site in ["AV-1", "AV-2", "AV-3"]:
            site_data = self.cleaner.wq[
                self.cleaner.wq["Site"] == site
            ].copy()

            site_data = site_data.sort_values("Date")

            trend = self.ml.trend_results[site]

            start_date = site_data["Date"].iloc[0]
            last_date = site_data["Date"].iloc[-1]

            slope = trend["slope"]
            intercept = trend["intercept"]

            # Observed records
            for _, row in site_data.iterrows():
                day = (row["Date"] - start_date).days

                result.append({
                    "Site": site,
                    "Date": row["Date"].strftime("%Y-%m-%d"),
                    "Day": day,
                    "DO_Value": float(row["DO"]),
                    "Trend_Value": round(
                        float(intercept + slope * day),
                        3
                    ),
                    "Projection_Value": None,
                    "Type": "Observed",
                    "Safe_Minimum": 7.0,
                })

            # 30-day projected records
            last_day = (last_date - start_date).days

            for i in range(0, 31):
                day = last_day + i
                projection_date = start_date + pd.Timedelta(days=day)

                result.append({
                    "Site": site,
                    "Date": projection_date.strftime("%Y-%m-%d"),
                    "Day": day,
                    "DO_Value": None,
                    "Trend_Value": None,
                    "Projection_Value": round(
                        float(intercept + slope * day),
                        3
                    ),
                    "Type": "Projection",
                    "Safe_Minimum": 7.0,
                })

        return result

    def count_sites_at_risk(self):
        """
        Count sites with high or critical risk.
        """

        count = 0

        for site in ["AV-1", "AV-2", "AV-3"]:
            if self.risk_level(site) in ["CRITICAL", "HIGH"]:
                count += 1

        return count

    def risk_level(self, site):
        """
        Return risk level for one site.
        """

        stats = self.cleaner.wq_stats.loc[site]

        do_min = stats["DO_min"]
        do_mean = stats["DO_mean"]

        if do_min < 6.5 or do_mean < 7.0:
            return "CRITICAL"

        if do_min < 7.0:
            return "HIGH"

        if do_mean < 7.5:
            return "MEDIUM"

        return "LOW"

    def status_label(self, risk):
        """
        Return plain-English dashboard status.
        """

        if risk == "CRITICAL":
            return "High priority"

        if risk == "HIGH":
            return "Monitor closely"

        if risk == "MEDIUM":
            return "Moderate"

        return "Good"

    def action(self, site):
        """
        Return suggested monitoring action.
        """

        risk = self.risk_level(site)

        if risk == "CRITICAL":
            return "Immediate intervention required"

        if risk == "HIGH":
            return "Urgent targeted monitoring"

        if risk == "MEDIUM":
            return "Regular scheduled monitoring"

        return "Maintain current conditions"