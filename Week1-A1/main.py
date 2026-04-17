import pandas as pd
import matplotlib.pyplot as plt


def main():
    # load the excel file
    file_path = "Data_set_w1A1.xlsx"
    sheet_name = "descriptive_aggregation (1)"
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # create new column for sales per unit
    # sales per unit = total sales / total quantity
    df["sales_per_unit"] = df["sales_sum"] / df["quantity_sum"]

    # show dataset
    print("Dataset:")
    print(df)
    print()

    # highest total sales
    top_sales = df.loc[df["sales_sum"].idxmax()]
    print("1. Category with highest total sales:")
    print(f"Category {top_sales['category']} has the highest total sales = {top_sales['sales_sum']}")
    print()

    # highest average sales
    top_avg_sales = df.loc[df["sales_mean"].idxmax()]
    print("2. Category with highest average sales:")
    print(f"Category {top_avg_sales['category']} has the highest average sales = {top_avg_sales['sales_mean']:.2f}")
    print()

    # highest sales count
    top_count = df.loc[df["sales_count"].idxmax()]
    print("3. Category with highest sales count:")
    print(f"Category {top_count['category']} has the highest sales count = {top_count['sales_count']}")
    print()

    # highest quantity sold
    top_quantity = df.loc[df["quantity_sum"].idxmax()]
    print("4. Category with highest quantity sold:")
    print(f"Category {top_quantity['category']} has the highest quantity sold = {top_quantity['quantity_sum']}")
    print()

    # highest sales per unit
    top_unit = df.loc[df["sales_per_unit"].idxmax()]
    print("5. Category with highest sales per unit:")
    print(f"Category {top_unit['category']} has the highest sales per unit = {top_unit['sales_per_unit']:.2f}")
    print()

    # weakest category by total sales
    weakest = df.loc[df["sales_sum"].idxmin()]
    print("6. Weakest category overall:")
    print(f"Category {weakest['category']} has the lowest total sales = {weakest['sales_sum']}")
    print()

    # overall story
    print("Overall Story:")
    print(
        f"Category {top_sales['category']} is the strongest category because it has the highest total sales, "
        f"highest average sales, highest sales count, highest quantity sold, and highest sales per unit. "
        f"Category {weakest['category']} performs the weakest overall because it has the lowest total sales."
    )

    # -----------------------------
    # graphs to show findings
    # -----------------------------
    fig, axes = plt.subplots(3, 2, figsize=(8, 6))

    # graph 1 - total sales
    axes[0, 0].bar(df["category"], df["sales_sum"])
    axes[0, 0].set_title("Total Sales by Category")
    axes[0, 0].set_xlabel("Category")
    axes[0, 0].set_ylabel("Sales Sum")

    # graph 2 - average sales
    axes[0, 1].bar(df["category"], df["sales_mean"])
    axes[0, 1].set_title("Average Sales by Category")
    axes[0, 1].set_xlabel("Category")
    axes[0, 1].set_ylabel("Sales Mean")

    # graph 3 - sales count
    axes[1, 0].bar(df["category"], df["sales_count"])
    axes[1, 0].set_title("Sales Count by Category")
    axes[1, 0].set_xlabel("Category")
    axes[1, 0].set_ylabel("Sales Count")

    # graph 4 - quantity sold
    axes[1, 1].bar(df["category"], df["quantity_sum"])
    axes[1, 1].set_title("Quantity Sold by Category")
    axes[1, 1].set_xlabel("Category")
    axes[1, 1].set_ylabel("Quantity Sum")

    # graph 5 - sales per unit
    axes[2, 0].bar(df["category"], df["sales_per_unit"])
    axes[2, 0].set_title("Sales per Unit by Category")
    axes[2, 0].set_xlabel("Category")
    axes[2, 0].set_ylabel("Sales per Unit")

    # remove empty last graph
    fig.delaxes(axes[2, 1])

    # fix spacing
    plt.tight_layout()

    # show all graphs
    plt.show()


if __name__ == "__main__":
    main()