import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load dataset"""
    return pd.read_csv(file_path)

def calculate_correlation(df):
    """Calculate correlation between Age and Net Worth"""
    print(df)
    correlation = df['Age'].corr(df['Net Worth'])
    return correlation

def plot_correlation(df):
    """Visualize correlation using scatter plot with regression line"""
    plt.figure()
    sns.regplot(x='Age', y='Net Worth', data=df)

    plt.title('Correlation between Age and Net Worth')
    plt.xlabel('Age')
    plt.ylabel('Net Worth')

    # save graph
    plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')

    plt.show()

def main():
    # File path
    file_path = 'age_networth.csv'

    # Load data
    df = load_data(file_path)
    print("Columns in dataset:", df.columns)

    # Display first few rows
    print("Dataset Preview:")
    print(df.head())

    # Calculate correlation
    correlation = calculate_correlation(df)
    print(f"\nCorrelation between Age and Net Worth: {correlation:.2f}")

    # Plot graph
    plot_correlation(df)

# Run the program
if __name__ == "__main__":
    main()