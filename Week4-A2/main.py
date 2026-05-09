"""
World Happiness SQL Analysis
This script runs SQL queries and saves the results.
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt


def load_dataset(file_name):
    """Load CSV file."""
    return pd.read_csv(file_name)


def create_database(df, db_name):
    """Create SQLite database."""
    conn = sqlite3.connect(db_name)

    # Save dataframe to SQL table
    df.to_sql("world_happiness", conn, index=False, if_exists="replace")

    return conn


def create_gdp_category_and_ranking_query():
    """SQL: GDP categories + avg happiness + ranking."""
    return """
    WITH gdp_categorised AS (
        SELECT
            Country,
            Happiness_Score,
            GDP_per_Capita,
            CASE
                WHEN GDP_per_Capita < 0.80 THEN 'Low GDP'
                WHEN GDP_per_Capita BETWEEN 0.80 AND 1.30 THEN 'Medium GDP'
                ELSE 'High GDP'
            END AS GDP_Category
        FROM world_happiness
    ),
    category_average AS (
        SELECT
            GDP_Category,
            ROUND(AVG(Happiness_Score), 2) AS Avg_Happiness
        FROM gdp_categorised
        GROUP BY GDP_Category
    ),
    country_ranking AS (
        SELECT
            Country,
            GDP_Category,
            GDP_per_Capita,
            Happiness_Score,
            RANK() OVER (
                PARTITION BY GDP_Category
                ORDER BY Happiness_Score DESC
            ) AS Rank_Within_Category
        FROM gdp_categorised
    )
    SELECT
        cr.GDP_Category,
        cr.Country,
        cr.GDP_per_Capita,
        cr.Happiness_Score,
        ca.Avg_Happiness,
        cr.Rank_Within_Category
    FROM country_ranking cr
    JOIN category_average ca
        ON cr.GDP_Category = ca.GDP_Category
    ORDER BY cr.GDP_Category, cr.Rank_Within_Category;
    """


def create_corruption_group_comparison_query():
    """SQL: compare high vs low corruption groups."""
    return """
    WITH corruption_groups AS (
        SELECT
            Country,
            Happiness_Score,
            GDP_per_Capita,
            Social_Support,
            Healthy_Life_Expectancy,
            Freedom_to_Make_Choices,
            Generosity,
            Perceptions_of_Corruption,
            CASE
                WHEN Perceptions_of_Corruption >= (
                    SELECT AVG(Perceptions_of_Corruption)
                    FROM world_happiness
                ) THEN 'High Corruption Perception'
                ELSE 'Low Corruption Perception'
            END AS Corruption_Group
        FROM world_happiness
    )
    SELECT
        Corruption_Group,
        COUNT(*) AS Country_Count,
        ROUND(AVG(Happiness_Score), 2) AS Avg_Happiness,
        ROUND(AVG(GDP_per_Capita), 2) AS Avg_GDP,
        ROUND(AVG(Social_Support), 2) AS Avg_Social_Support,
        ROUND(AVG(Healthy_Life_Expectancy), 2) AS Avg_Healthy_Life_Expectancy,
        ROUND(AVG(Freedom_to_Make_Choices), 2) AS Avg_Freedom,
        ROUND(AVG(Generosity), 2) AS Avg_Generosity,
        ROUND(AVG(Perceptions_of_Corruption), 2) AS Avg_Corruption_Perception,
        ROUND(AVG(Happiness_Score) - (
            SELECT AVG(Happiness_Score)
            FROM world_happiness
        ), 2) AS Difference_From_Overall_Happiness
    FROM corruption_groups
    GROUP BY Corruption_Group
    ORDER BY Avg_Happiness DESC;
    """


def run_query(conn, query):
    """Run SQL query."""
    return pd.read_sql_query(query, conn)


def save_table_image(df, title, file_name, font_size=8):
    """Save table as image."""
    fig, ax = plt.subplots(figsize=(14, max(2.5, len(df) * 0.35)))

    ax.axis("off")  # hide axis
    ax.set_title(title, fontsize=14, pad=14)

    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(query1, query2, result1, result2):
    """Save SQL, CSV, and images."""

    # Save SQL file
    with open("queries.sql", "w", encoding="utf-8") as f:
        f.write(query1 + "\n\n" + query2)

    # Save CSV results
    result1.to_csv("query1_results.csv", index=False)
    result2.to_csv("query2_results.csv", index=False)

    # Save images
    save_table_image(
        result1,
        "Query 1: GDP Category and Ranking",
        "query1_results_screenshot.png"
    )

    save_table_image(
        result2,
        "Query 2: Corruption Comparison",
        "query2_results_screenshot.png"
    )


def main():
    """Main function."""

    # Files in same folder
    csv_file = "world_happiness_dataset.csv"
    db_file = "world_happiness.db"

    # Load data
    df = load_dataset(csv_file)

    # Create DB
    conn = create_database(df, db_file)

    # Create queries
    query1 = create_gdp_category_and_ranking_query()
    query2 = create_corruption_group_comparison_query()

    # Run queries
    result1 = run_query(conn, query1)
    result2 = run_query(conn, query2)

    # Show results
    print("\nQuery 1 Result:")
    print(result1)

    print("\nQuery 2 Result:")
    print(result2)

    # Save outputs
    save_outputs(query1, query2, result1, result2)

    # Close DB
    conn.close()

    print("\nFiles created in same folder.")


if __name__ == "__main__":
    main()