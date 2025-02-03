import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df):
    """
    Plots the correlation matrix.
    """
    plt.figure(figsize=(12, 10))

    corr = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.tight_layout()
    plt.savefig('data/processed/correlation_matrix.png')
    plt.close()

def plot_distribution(df, column):
    """
    Plots the distribution of a specified column.
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig(f'data/processed/{column}_distribution.png')
    plt.close()
