import os
import pandas as pd
import matplotlib.pyplot as plt


WATER_FILE = "water_quality.csv"
FISH_FILE = "fish_population.csv"
OUTPUT_FOLDER = "charts"


def create_output_folder():
    # make folder for saving charts
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)


def clean_column_names(df):
    # make column names easy to use
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("°", "", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def load_data():
    # read csv files
    water_df = pd.read_csv(WATER_FILE)
    fish_df = pd.read_csv(FISH_FILE)

    return water_df, fish_df


def clean_water_data(water_df):
    # clean water quality data
    water_df = clean_column_names(water_df)
    water_df = water_df.drop_duplicates()

    water_df["Date"] = pd.to_datetime(water_df["Date"], errors="coerce")

    water_df["Temperature_C"] = pd.to_numeric(
        water_df["Temperature_C"],
        errors="coerce"
    )

    water_df["pH"] = pd.to_numeric(
        water_df["pH"],
        errors="coerce"
    )

    water_df["Dissolved_Oxygen_mg_L"] = pd.to_numeric(
        water_df["Dissolved_Oxygen_mg_L"],
        errors="coerce"
    )

    water_df = water_df.dropna(
        subset=[
            "Site_ID",
            "Date",
            "Temperature_C",
            "pH",
            "Dissolved_Oxygen_mg_L"
        ]
    )

    return water_df


def clean_fish_data(fish_df):
    # clean fish population data
    fish_df = clean_column_names(fish_df)
    fish_df = fish_df.drop_duplicates()

    fish_df["Date"] = pd.to_datetime(fish_df["Date"], errors="coerce")

    fish_df["Count"] = pd.to_numeric(
        fish_df["Count"],
        errors="coerce"
    )

    fish_df["Average_Size_cm"] = pd.to_numeric(
        fish_df["Average_Size_cm"],
        errors="coerce"
    )

    # no fish observed is made 0
    fish_df["Count"] = fish_df["Count"].fillna(0)
    fish_df["Average_Size_cm"] = fish_df["Average_Size_cm"].fillna(0)

    fish_df = fish_df.dropna(
        subset=[
            "Site_ID",
            "Date",
            "Species"
        ]
    )

    return fish_df


def show_data_quality(water_df, fish_df):
    # show if still have missing values
    print("\nMissing Values - Water Quality Data")
    print(water_df.isnull().sum())

    print("\nMissing Values - Fish Population Data")
    print(fish_df.isnull().sum())

    print("\nWater Data Shape:", water_df.shape)
    print("Fish Data Shape:", fish_df.shape)


def analyse_water_data(water_df):
    # analyse water dataset only
    print("\nWater Quality Summary")
    print(water_df.describe())

    print("\nAverage Water Quality by Site")
    print(
        water_df.groupby("Site_ID")[
            ["Temperature_C", "pH", "Dissolved_Oxygen_mg_L"]
        ].mean()
    )


def analyse_fish_data(fish_df):
    # analyse fish dataset only
    print("\nFish Population Summary")
    print(fish_df.describe())

    print("\nFish Count by Site")
    print(
        fish_df.groupby("Site_ID")["Count"].sum()
    )

    print("\nSpecies Recorded")
    print(fish_df["Species"].value_counts())


def merge_for_relationship(water_df, fish_df):
    # combine only when checking relationship
    relationship_df = pd.merge(
        water_df,
        fish_df,
        on=["Site_ID", "Date"],
        how="inner"
    )

    return relationship_df


def analyse_relationship(relationship_df):
    # correlation analysis for water quality and fish
    print("\nMerged Dataset for Relationship Analysis")
    print(relationship_df)

    correlation = relationship_df[
        [
            "Temperature_C",
            "pH",
            "Dissolved_Oxygen_mg_L",
            "Count",
            "Average_Size_cm"
        ]
    ].corr()

    print("\nCorrelation Matrix")
    print(correlation)

    return correlation


def plot_dissolved_oxygen(water_df):
    # figure 1 for challenge 1
    plt.figure(figsize=(8, 5))

    for site in water_df["Site_ID"].unique():
        site_data = water_df[water_df["Site_ID"] == site]

        plt.plot(
            site_data["Date"],
            site_data["Dissolved_Oxygen_mg_L"],
            marker="o",
            label=site
        )

    plt.title("Dissolved Oxygen Levels by Site")
    plt.xlabel("Date")
    plt.ylabel("Dissolved Oxygen (mg/L)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "figure_1_dissolved_oxygen_by_site.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_fish_count(fish_df):
    # figure 2 for challenge 2
    plt.figure(figsize=(8, 5))

    labels = (
        fish_df["Site_ID"]
        + " "
        + fish_df["Date"].dt.strftime("%b-%Y")
    )

    plt.bar(labels, fish_df["Count"])

    plt.title("Fish Count by Site and Date")
    plt.xlabel("Site and Date")
    plt.ylabel("Fish Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "figure_2_fish_count_by_site.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_oxygen_vs_fish(relationship_df):
    # figure 3 for relationship analysis
    plt.figure(figsize=(8, 5))

    plt.scatter(
        relationship_df["Dissolved_Oxygen_mg_L"],
        relationship_df["Count"]
    )

    for _, row in relationship_df.iterrows():
        label = row["Site_ID"] + " " + row["Date"].strftime("%b")
        plt.annotate(
            label,
            (row["Dissolved_Oxygen_mg_L"], row["Count"]),
            fontsize=8
        )

    plt.title("Relationship Between Dissolved Oxygen and Fish Count")
    plt.xlabel("Dissolved Oxygen (mg/L)")
    plt.ylabel("Fish Count")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "figure_3_oxygen_vs_fish_count.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_temperature(water_df):
    # figure 4 for water temperature
    plt.figure(figsize=(8, 5))

    for site in water_df["Site_ID"].unique():
        site_data = water_df[water_df["Site_ID"] == site]

        plt.plot(
            site_data["Date"],
            site_data["Temperature_C"],
            marker="o",
            label=site
        )

    plt.title("Water Temperature by Site")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "figure_4_temperature_by_site.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def plot_species_distribution(fish_df):
    # figure 5 for species distribution
    plt.figure(figsize=(8, 5))

    species_count = fish_df["Species"].value_counts()

    plt.bar(species_count.index, species_count.values)

    plt.title("Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_FOLDER, "figure_5_species_distribution.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


def generate_findings(water_df, fish_df):
    # simple findings for report
    lowest_oxygen = water_df.loc[
        water_df["Dissolved_Oxygen_mg_L"].idxmin()
    ]

    lowest_fish = fish_df.loc[
        fish_df["Count"].idxmin()
    ]

    print("\nKey Findings")

    print(
        f"Lowest dissolved oxygen was recorded at "
        f"{lowest_oxygen['Site_ID']} on "
        f"{lowest_oxygen['Date'].date()} with "
        f"{lowest_oxygen['Dissolved_Oxygen_mg_L']} mg/L."
    )

    print(
        f"Lowest fish count was recorded at "
        f"{lowest_fish['Site_ID']} on "
        f"{lowest_fish['Date'].date()} with "
        f"{lowest_fish['Count']} fish observed."
    )

    print(
        "\nInterpretation: AV-3 recorded the lowest dissolved oxygen "
        "and also recorded no fish in November. This may show that "
        "low dissolved oxygen can affect fish population health."
    )


def main():
    create_output_folder()

    # load data
    water_df, fish_df = load_data()

    # clean data separately first
    water_df = clean_water_data(water_df)
    fish_df = clean_fish_data(fish_df)

    # check data quality
    show_data_quality(water_df, fish_df)

    # analyse datasets separately
    analyse_water_data(water_df)
    analyse_fish_data(fish_df)

    # merge only for relationship analysis
    relationship_df = merge_for_relationship(water_df, fish_df)

    # save cleaned and merged files
    water_df.to_csv("cleaned_water_quality.csv", index=False)
    fish_df.to_csv("cleaned_fish_population.csv", index=False)
    relationship_df.to_csv("relationship_analysis_data.csv", index=False)

    # relationship analysis
    analyse_relationship(relationship_df)

    # create all charts
    plot_dissolved_oxygen(water_df)
    plot_fish_count(fish_df)
    plot_oxygen_vs_fish(relationship_df)
    plot_temperature(water_df)
    plot_species_distribution(fish_df)

    # print findings
    generate_findings(water_df, fish_df)


if __name__ == "__main__":
    main()