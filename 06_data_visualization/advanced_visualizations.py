"""
Advanced Visualizations

This script demonstrates:
- Faceted plots
- Time series visualization
- Statistical annotations
- Complex multi-plot figures
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


def faceted_plots():
    """Demonstrate faceted plots with FacetGrid"""
    print("=" * 50)
    print("FACETED PLOTS")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    data = []
    for category in ['A', 'B', 'C']:
        for subcategory in ['X', 'Y']:
            n = 50
            data.extend([{
                'value': np.random.normal(loc=ord(category)-64, scale=1),
                'category': category,
                'subcategory': subcategory
            } for _ in range(n)])
    
    df = pd.DataFrame(data)
    
    # Create FacetGrid
    g = sns.FacetGrid(df, col='category', row='subcategory', height=3, aspect=1.2)
    g.map(sns.histplot, 'value', kde=True, color='skyblue', bins=15)
    g.set_axis_labels('Value', 'Count')
    g.set_titles(col_template='Category: {col_name}', row_template='Subcategory: {row_name}')
    g.fig.suptitle('Faceted Histogram Grid', y=1.02, fontsize=16)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'faceted_plots.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Faceted plots saved to: {output_path}")
    plt.close()


def time_series_visualization():
    """Demonstrate time series visualization"""
    print("\n" + "=" * 50)
    print("TIME SERIES VISUALIZATION")
    print("=" * 50)
    
    # Create time series data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Multiple time series with trends and seasonality
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    noise = np.random.randn(len(dates)) * 5
    
    df = pd.DataFrame({
        'date': dates,
        'sales': trend + seasonal + noise,
        'revenue': (trend + seasonal + noise) * 2.5 + np.random.randn(len(dates)) * 10
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Line plot with moving average
    axes[0].plot(df['date'], df['sales'], label='Daily Sales', alpha=0.6)
    axes[0].plot(df['date'], df['sales'].rolling(window=30).mean(), 
                label='30-Day Moving Average', linewidth=2, color='red')
    axes[0].set_title('Sales Over Time with Moving Average', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Sales')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Area plot
    axes[1].fill_between(df['date'], df['revenue'], alpha=0.4, color='green')
    axes[1].plot(df['date'], df['revenue'], color='darkgreen', linewidth=1)
    axes[1].set_title('Revenue Over Time (Area Plot)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Revenue')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Multiple series comparison
    axes[2].plot(df['date'], df['sales'], label='Sales', linewidth=2)
    axes[2].plot(df['date'], df['revenue'] / 2.5, label='Revenue (scaled)', linewidth=2)
    axes[2].set_title('Sales vs Revenue Comparison', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'time_series.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Time series plots saved to: {output_path}")
    plt.close()


def statistical_annotations():
    """Demonstrate plots with statistical annotations"""
    print("\n" + "=" * 50)
    print("STATISTICAL ANNOTATIONS")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'group': ['A']*50 + ['B']*50 + ['C']*50,
        'value': np.concatenate([
            np.random.normal(100, 10, 50),
            np.random.normal(110, 12, 50),
            np.random.normal(105, 8, 50)
        ])
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Box plot with annotations
    bp = axes[0].boxplot([df[df['group']=='A']['value'],
                          df[df['group']=='B']['value'],
                          df[df['group']=='C']['value']],
                         labels=['Group A', 'Group B', 'Group C'],
                         patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add mean markers
    means = [df[df['group']==g]['value'].mean() for g in ['A', 'B', 'C']]
    axes[0].plot(range(1, 4), means, 'r*', markersize=15, label='Mean')
    
    # Add annotations
    for i, (group, mean) in enumerate(zip(['A', 'B', 'C'], means), 1):
        axes[0].annotate(f'μ={mean:.1f}', xy=(i, mean), 
                        xytext=(i+0.3, mean), fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    axes[0].set_title('Box Plot with Statistical Annotations', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Violin plot with quartiles
    parts = axes[1].violinplot([df[df['group']=='A']['value'],
                                df[df['group']=='B']['value'],
                                df[df['group']=='C']['value']],
                               positions=[1, 2, 3],
                               showmeans=True, showmedians=True)
    
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['Group A', 'Group B', 'Group C'])
    axes[1].set_title('Violin Plot with Quartiles', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'statistical_annotations.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Statistical annotations saved to: {output_path}")
    plt.close()


def complex_dashboard():
    """Create a complex dashboard with multiple plot types"""
    print("\n" + "=" * 50)
    print("COMPLEX DASHBOARD")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'value': np.random.uniform(10, 100, n)
    })
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Large scatter plot (top left, 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    for cat in ['A', 'B', 'C']:
        data = df[df['category'] == cat]
        ax1.scatter(data['x'], data['y'], s=data['value'], 
                   alpha=0.6, label=f'Category {cat}')
    ax1.set_title('Scatter Plot by Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['value'], bins=20, color='skyblue', edgecolor='black')
    ax2.set_title('Value Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Box plot (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    df.boxplot(column='value', by='category', ax=ax3)
    ax3.set_title('Value by Category', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Value')
    plt.sca(ax3)
    plt.xticks(rotation=0)
    
    # Bar plot (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    category_means = df.groupby('category')['value'].mean()
    ax4.bar(category_means.index, category_means.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('Average Value by Category', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Category')
    ax4.set_ylabel('Average Value')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Pie chart (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    category_counts = df['category'].value_counts()
    ax5.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
           colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax5.set_title('Category Distribution', fontsize=12, fontweight='bold')
    
    # Summary statistics (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
    Summary Statistics:
    
    Total Samples: {len(df)}
    
    Value Statistics:
      Mean: {df['value'].mean():.2f}
      Median: {df['value'].median():.2f}
      Std Dev: {df['value'].std():.2f}
      Min: {df['value'].min():.2f}
      Max: {df['value'].max():.2f}
    
    Category Counts:
      A: {(df['category']=='A').sum()}
      B: {(df['category']=='B').sum()}
      C: {(df['category']=='C').sum()}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    fig.suptitle('Comprehensive Data Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    output_path = os.path.join(OUTPUT_DIR, 'complex_dashboard.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Complex dashboard saved to: {output_path}")
    plt.close()


def correlation_visualization():
    """Visualize correlations in different ways"""
    print("\n" + "=" * 50)
    print("CORRELATION VISUALIZATION")
    print("=" * 50)
    
    # Create correlated data
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = 0.8 * x1 + 0.2 * np.random.randn(n)
    x3 = -0.6 * x1 + 0.4 * np.random.randn(n)
    x4 = np.random.randn(n)
    
    df = pd.DataFrame({
        'Variable 1': x1,
        'Variable 2': x2,
        'Variable 3': x3,
        'Variable 4': x4
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Correlation heatmap
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
               square=True, linewidths=2, cbar_kws={'label': 'Correlation'},
               ax=axes[0])
    axes[0].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    
    # Clustermap (hierarchical clustering)
    # Create separate figure for clustermap
    g = sns.clustermap(df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                      square=True, linewidths=2, figsize=(8, 6),
                      cbar_kws={'label': 'Correlation'})
    g.fig.suptitle('Clustered Correlation Heatmap', y=1.02, fontsize=14, fontweight='bold')
    
    # Save correlation heatmap
    output_path1 = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
    axes[0].figure.savefig(output_path1, dpi=100, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {output_path1}")
    plt.close(axes[0].figure)
    
    # Save clustermap
    output_path2 = os.path.join(OUTPUT_DIR, 'correlation_clustermap.png')
    g.savefig(output_path2, dpi=100, bbox_inches='tight')
    print(f"Correlation clustermap saved to: {output_path2}")
    plt.close()


def main():
    """Main function to run all examples"""
    print("Welcome to Advanced Visualizations!\n")
    
    # Run all demonstrations
    faceted_plots()
    time_series_visualization()
    statistical_annotations()
    complex_dashboard()
    correlation_visualization()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print(f"All plots saved in: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
