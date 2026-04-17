import pandas as pd
import matplotlib.pyplot as plt


def main():
    # load dataset
    df = pd.read_csv("Housing.csv")

    # create area size groups (small, medium, large)
    df["area_group"] = pd.cut(
        df["area"],
        bins=3,
        labels=["Small", "Medium", "Large"]
    )

    # group data and get average price
    bedroom_avg = df.groupby("bedrooms")["price"].mean()
    bathroom_avg = df.groupby("bathrooms")["price"].mean()
    parking_avg = df.groupby("parking")["price"].mean()
    area_avg = df.groupby("area_group")["price"].mean()

    # print findings
    print("Findings:")
    print("1. Price generally increases as bedrooms increase, but not always consistent.")
    print("2. More bathrooms and parking spaces increase the price.")
    print("3. Houses with larger area size have higher average price.")
    print("4. Even with same bedrooms, price changes based on area size.")
    print()

    print("Overall Story:")
    print("House price is affected by multiple factors.")
    print("Bedrooms, bathrooms, and parking increase price,")
    print("but area size also plays a big role.")
    print("This shows price is not determined by bedrooms alone.")
    print()

    # -----------------------------
    # main graphs (2x2)
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # graph 1 - bedrooms
    bedroom_avg.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_title("Price by Bedrooms")
    axes[0, 0].set_xlabel("Bedrooms")
    axes[0, 0].set_ylabel("Avg Price")

    # graph 2 - bathrooms
    bathroom_avg.plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title("Price by Bathrooms")
    axes[0, 1].set_xlabel("Bathrooms")
    axes[0, 1].set_ylabel("Avg Price")

    # graph 3 - parking
    parking_avg.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Price by Parking")
    axes[1, 0].set_xlabel("Parking")
    axes[1, 0].set_ylabel("Avg Price")

    # graph 4 - area size (UPDATED)
    area_avg.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Price by Area Size")
    axes[1, 1].set_xlabel("Area Size")
    axes[1, 1].set_ylabel("Avg Price")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # evidence graph (important)
    # -----------------------------
    bedroom_area_avg = df.groupby(["bedrooms", "area_group"])["price"].mean().unstack()

    plt.figure(figsize=(8, 5))

    for col in bedroom_area_avg.columns:
        plt.plot(
            bedroom_area_avg.index,
            bedroom_area_avg[col],
            marker='o',
            label=f"Area: {col}"
        )

    plt.title("Bedrooms vs Price (Area Size Effect)")
    plt.xlabel("Bedrooms")
    plt.ylabel("Avg Price")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()