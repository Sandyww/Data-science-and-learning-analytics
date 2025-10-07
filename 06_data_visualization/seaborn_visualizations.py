"""
Seaborn Visualizations

This script demonstrates:
- Distribution plots
- Categorical plots
- Relationship plots
- Heatmaps
- Styling and themes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seaborn style
sns.set_theme(style="whitegrid")


def create_sample_data():
    """Create sample datasets for visualization"""
    np.random.seed(42)
    
    # Dataset 1: Tips-like data
    tips_data = {
        'total_bill': np.random.uniform(10, 50, 100),
        'tip': np.random.uniform(1, 10, 100),
        'sex': np.random.choice(['Male', 'Female'], 100),
        'smoker': np.random.choice(['Yes', 'No'], 100),
        'day': np.random.choice(['Thur', 'Fri', 'Sat', 'Sun'], 100),
        'time': np.random.choice(['Lunch', 'Dinner'], 100),
        'size': np.random.randint(1, 7, 100)
    }
    df_tips = pd.DataFrame(tips_data)
    
    # Dataset 2: Iris-like data
    n_samples = 150
    iris_data = {
        'sepal_length': np.concatenate([
            np.random.normal(5.0, 0.5, 50),
            np.random.normal(6.0, 0.5, 50),
            np.random.normal(6.5, 0.5, 50)
        ]),
        'sepal_width': np.concatenate([
            np.random.normal(3.4, 0.3, 50),
            np.random.normal(2.8, 0.3, 50),
            np.random.normal(3.0, 0.3, 50)
        ]),
        'petal_length': np.concatenate([
            np.random.normal(1.5, 0.2, 50),
            np.random.normal(4.2, 0.5, 50),
            np.random.normal(5.5, 0.5, 50)
        ]),
        'petal_width': np.concatenate([
            np.random.normal(0.2, 0.1, 50),
            np.random.normal(1.3, 0.2, 50),
            np.random.normal(2.0, 0.3, 50)
        ]),
        'species': ['setosa']*50 + ['versicolor']*50 + ['virginica']*50
    }
    df_iris = pd.DataFrame(iris_data)
    
    return df_tips, df_iris


def distribution_plots(df):
    """Demonstrate distribution plots"""
    print("=" * 50)
    print("DISTRIBUTION PLOTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram with KDE
    sns.histplot(data=df, x='total_bill', kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Histogram with KDE')
    
    # KDE plot
    sns.kdeplot(data=df, x='total_bill', hue='sex', ax=axes[0, 1])
    axes[0, 1].set_title('KDE Plot by Gender')
    
    # Box plot
    sns.boxplot(data=df, x='day', y='total_bill', ax=axes[1, 0])
    axes[1, 0].set_title('Box Plot by Day')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Violin plot
    sns.violinplot(data=df, x='day', y='total_bill', hue='sex', ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot by Day and Gender')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_distributions.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Distribution plots saved to: {output_path}")
    plt.close()


def categorical_plots(df):
    """Demonstrate categorical plots"""
    print("\n" + "=" * 50)
    print("CATEGORICAL PLOTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Bar plot
    sns.barplot(data=df, x='day', y='total_bill', hue='sex', ax=axes[0, 0])
    axes[0, 0].set_title('Bar Plot')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Count plot
    sns.countplot(data=df, x='day', hue='smoker', ax=axes[0, 1])
    axes[0, 1].set_title('Count Plot')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Point plot
    sns.pointplot(data=df, x='day', y='total_bill', hue='sex', ax=axes[1, 0])
    axes[1, 0].set_title('Point Plot')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Strip plot
    sns.stripplot(data=df, x='day', y='total_bill', hue='sex', dodge=True, ax=axes[1, 1])
    axes[1, 1].set_title('Strip Plot')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_categorical.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Categorical plots saved to: {output_path}")
    plt.close()


def relationship_plots(df_iris):
    """Demonstrate relationship plots"""
    print("\n" + "=" * 50)
    print("RELATIONSHIP PLOTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    sns.scatterplot(data=df_iris, x='sepal_length', y='sepal_width', 
                   hue='species', style='species', s=100, ax=axes[0])
    axes[0].set_title('Scatter Plot with Hue and Style')
    
    # Line plot with confidence interval
    df_time = pd.DataFrame({
        'time': range(10),
        'value': np.cumsum(np.random.randn(10))
    })
    sns.lineplot(data=df_time, x='time', y='value', marker='o', ax=axes[1])
    axes[1].set_title('Line Plot')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_relationships.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Relationship plots saved to: {output_path}")
    plt.close()


def pairplot_demo(df_iris):
    """Demonstrate pair plot"""
    print("\n" + "=" * 50)
    print("PAIR PLOT")
    print("=" * 50)
    
    # Create pair plot
    g = sns.pairplot(df_iris, hue='species', diag_kind='kde', 
                     palette='Set2', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pair Plot of Iris Dataset', y=1.02, fontsize=16)
    
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_pairplot.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Pair plot saved to: {output_path}")
    plt.close()


def heatmap_demo():
    """Demonstrate heatmaps"""
    print("\n" + "=" * 50)
    print("HEATMAPS")
    print("=" * 50)
    
    # Create correlation matrix
    np.random.seed(42)
    data = np.random.randn(10, 12)
    df = pd.DataFrame(data, columns=[f'Feature{i}' for i in range(12)])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=1, ax=axes[0])
    axes[0].set_title('Correlation Heatmap')
    
    # Regular heatmap
    sns.heatmap(df.iloc[:8, :8], annot=True, fmt='.1f', cmap='YlGnBu', 
                ax=axes[1], cbar_kws={'label': 'Value'})
    axes[1].set_title('Annotated Heatmap')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_heatmaps.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Heatmaps saved to: {output_path}")
    plt.close()


def regression_plot(df):
    """Demonstrate regression plots"""
    print("\n" + "=" * 50)
    print("REGRESSION PLOTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regression plot
    sns.regplot(data=df, x='total_bill', y='tip', ax=axes[0], 
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    axes[0].set_title('Regression Plot')
    
    # Residual plot
    sns.residplot(data=df, x='total_bill', y='tip', ax=axes[1], 
                 scatter_kws={'alpha': 0.5})
    axes[1].set_title('Residual Plot')
    axes[1].axhline(y=0, color='red', linestyle='--')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_regression.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Regression plots saved to: {output_path}")
    plt.close()


def style_themes_demo():
    """Demonstrate different styles and themes"""
    print("\n" + "=" * 50)
    print("STYLES AND THEMES")
    print("=" * 50)
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    styles = ['darkgrid', 'whitegrid', 'dark', 'white']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, style in enumerate(styles):
        sns.set_style(style)
        ax = axes[idx]
        ax.plot(x, y, linewidth=2)
        ax.set_title(f'Style: {style}', fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_styles.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Style themes saved to: {output_path}")
    plt.close()
    
    # Reset to default
    sns.set_theme(style="whitegrid")


def color_palettes_demo():
    """Demonstrate color palettes"""
    print("\n" + "=" * 50)
    print("COLOR PALETTES")
    print("=" * 50)
    
    palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, palette in enumerate(palettes):
        colors = sns.color_palette(palette, 5)
        ax = axes[idx]
        ax.barh(range(5), [1]*5, color=colors)
        ax.set_title(f'Palette: {palette}', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'seaborn_palettes.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Color palettes saved to: {output_path}")
    plt.close()


def main():
    """Main function to run all examples"""
    print("Welcome to Seaborn Visualizations!\n")
    
    # Create sample data
    df_tips, df_iris = create_sample_data()
    
    # Run all demonstrations
    distribution_plots(df_tips)
    categorical_plots(df_tips)
    relationship_plots(df_iris)
    pairplot_demo(df_iris)
    heatmap_demo()
    regression_plot(df_tips)
    style_themes_demo()
    color_palettes_demo()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print(f"All plots saved in: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
