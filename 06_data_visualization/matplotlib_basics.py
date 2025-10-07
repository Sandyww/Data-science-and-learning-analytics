"""
Matplotlib Basics

This script demonstrates:
- Basic plot types
- Plot customization
- Subplots and layouts
- Saving figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def line_plots():
    """Demonstrate line plots"""
    print("=" * 50)
    print("LINE PLOTS")
    print("=" * 50)
    
    # Create data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(x, y1, label='sin(x)', linewidth=2, color='blue')
    plt.plot(x, y2, label='cos(x)', linewidth=2, color='red', linestyle='--')
    
    # Customize
    plt.title('Sine and Cosine Functions', fontsize=16, fontweight='bold')
    plt.xlabel('X axis', fontsize=12)
    plt.ylabel('Y axis', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'line_plot.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Line plot saved to: {output_path}")
    plt.close()


def bar_plots():
    """Demonstrate bar plots"""
    print("\n" + "=" * 50)
    print("BAR PLOTS")
    print("=" * 50)
    
    # Create data
    categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    values = [25, 40, 30, 55, 45]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Vertical bar plot
    ax1.bar(categories, values, color='skyblue', edgecolor='navy')
    ax1.set_title('Vertical Bar Plot', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Categories')
    ax1.set_ylabel('Values')
    ax1.grid(axis='y', alpha=0.3)
    
    # Horizontal bar plot
    ax2.barh(categories, values, color='lightcoral', edgecolor='darkred')
    ax2.set_title('Horizontal Bar Plot', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Categories')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'bar_plots.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Bar plots saved to: {output_path}")
    plt.close()


def scatter_plots():
    """Demonstrate scatter plots"""
    print("\n" + "=" * 50)
    print("SCATTER PLOTS")
    print("=" * 50)
    
    # Create data
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y = 2 * x + np.random.randn(n) * 0.5
    colors = np.random.rand(n)
    sizes = np.random.randint(20, 200, n)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis', edgecolors='black')
    
    # Customize
    plt.colorbar(scatter, label='Color Scale')
    plt.title('Scatter Plot with Variable Size and Color', fontsize=16, fontweight='bold')
    plt.xlabel('X axis', fontsize=12)
    plt.ylabel('Y axis', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'scatter_plot.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_path}")
    plt.close()


def histogram_plots():
    """Demonstrate histograms"""
    print("\n" + "=" * 50)
    print("HISTOGRAMS")
    print("=" * 50)
    
    # Create data
    np.random.seed(42)
    data1 = np.random.normal(100, 15, 1000)
    data2 = np.random.normal(120, 20, 1000)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Overlapping histograms
    plt.hist(data1, bins=30, alpha=0.6, label='Dataset 1', color='blue', edgecolor='black')
    plt.hist(data2, bins=30, alpha=0.6, label='Dataset 2', color='red', edgecolor='black')
    
    # Customize
    plt.title('Overlapping Histograms', fontsize=16, fontweight='bold')
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'histogram.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Histogram saved to: {output_path}")
    plt.close()


def pie_charts():
    """Demonstrate pie charts"""
    print("\n" + "=" * 50)
    print("PIE CHARTS")
    print("=" * 50)
    
    # Create data
    sizes = [30, 25, 20, 15, 10]
    labels = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
    colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'plum']
    explode = (0.1, 0, 0, 0, 0)  # Explode first slice
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Pie chart
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    plt.title('Pie Chart Distribution', fontsize=16, fontweight='bold')
    
    # Equal aspect ratio ensures circular pie
    plt.axis('equal')
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, 'pie_chart.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Pie chart saved to: {output_path}")
    plt.close()


def subplots_demo():
    """Demonstrate subplots"""
    print("\n" + "=" * 50)
    print("SUBPLOTS")
    print("=" * 50)
    
    # Create data
    x = np.linspace(0, 10, 100)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Line plot
    axes[0, 0].plot(x, np.sin(x), 'b-')
    axes[0, 0].set_title('Sine Wave')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('sin(x)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[0, 1].scatter(x, np.cos(x), c=x, cmap='viridis')
    axes[0, 1].set_title('Cosine Scatter')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('cos(x)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Bar plot
    categories = ['A', 'B', 'C', 'D']
    values = [3, 7, 2, 5]
    axes[1, 0].bar(categories, values, color='coral')
    axes[1, 0].set_title('Bar Chart')
    axes[1, 0].set_xlabel('Categories')
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Histogram
    data = np.random.randn(1000)
    axes[1, 1].hist(data, bins=30, color='green', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Histogram')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'subplots.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Subplots saved to: {output_path}")
    plt.close()


def customization_demo():
    """Demonstrate plot customization"""
    print("\n" + "=" * 50)
    print("PLOT CUSTOMIZATION")
    print("=" * 50)
    
    # Create data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Create figure with custom style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with customization
    ax.plot(x, y, linewidth=3, color='#FF6B6B', label='sin(x)')
    ax.fill_between(x, y, alpha=0.3, color='#FF6B6B')
    
    # Customize axes
    ax.set_title('Customized Plot', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('X axis', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y axis', fontsize=14, fontweight='bold')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add legend
    ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'customized_plot.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Customized plot saved to: {output_path}")
    plt.close()
    
    # Reset style
    plt.style.use('default')


def main():
    """Main function to run all examples"""
    print("Welcome to Matplotlib Basics!\n")
    
    # Run all demonstrations
    line_plots()
    bar_plots()
    scatter_plots()
    histogram_plots()
    pie_charts()
    subplots_demo()
    customization_demo()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print(f"All plots saved in: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
