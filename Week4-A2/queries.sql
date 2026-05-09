
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
    