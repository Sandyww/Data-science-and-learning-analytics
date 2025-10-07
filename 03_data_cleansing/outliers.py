"""
Outlier Detection and Treatment

This script demonstrates:
- Statistical methods for outlier detection (IQR, Z-score)
- Visualization techniques
- Treatment strategies
- Impact assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_sample_data():
    """Create sample data with outliers"""
    print("=" * 50)
    print("CREATING SAMPLE DATA WITH OUTLIERS")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Normal data
    normal_data = np.random.normal(100, 15, 95)
    
    # Add outliers
    outliers = np.array([200, 210, 220, 10, 5])
    
    # Combine
    data = np.concatenate([normal_data, outliers])
    np.random.shuffle(data)
    
    df = pd.DataFrame({'Value': data})
    
    print(f"\nCreated dataset with {len(df)} observations")
    print(f"Mean: {df['Value'].mean():.2f}")
    print(f"Median: {df['Value'].median():.2f}")
    print(f"Std: {df['Value'].std():.2f}")
    print(f"\nFirst 10 values:\n{df.head(10)}")
    
    return df


def iqr_method(df):
    """Detect outliers using Interquartile Range (IQR) method"""
    print("\n" + "=" * 50)
    print("IQR METHOD FOR OUTLIER DETECTION")
    print("=" * 50)
    
    # Calculate Q1, Q3, and IQR
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    
    print(f"\nQ1 (25th percentile): {Q1:.2f}")
    print(f"Q3 (75th percentile): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nOutlier bounds:")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    
    # Identify outliers
    outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]
    print(f"\nNumber of outliers detected: {len(outliers)}")
    print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")
    
    if len(outliers) > 0:
        print("\nOutlier values:")
        print(outliers.sort_values('Value'))
    
    # Mark outliers
    df['Is_Outlier_IQR'] = ((df['Value'] < lower_bound) | (df['Value'] > upper_bound))
    
    return df, lower_bound, upper_bound


def z_score_method(df):
    """Detect outliers using Z-score method"""
    print("\n" + "=" * 50)
    print("Z-SCORE METHOD FOR OUTLIER DETECTION")
    print("=" * 50)
    
    # Calculate Z-scores
    mean = df['Value'].mean()
    std = df['Value'].std()
    df['Z_Score'] = (df['Value'] - mean) / std
    
    print(f"\nMean: {mean:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    
    # Define threshold (typically 3)
    threshold = 3
    print(f"Threshold: {threshold}")
    
    # Identify outliers
    outliers = df[np.abs(df['Z_Score']) > threshold]
    print(f"\nNumber of outliers detected: {len(outliers)}")
    print(f"Percentage of outliers: {(len(outliers)/len(df))*100:.2f}%")
    
    if len(outliers) > 0:
        print("\nOutlier values and their Z-scores:")
        print(outliers[['Value', 'Z_Score']].sort_values('Z_Score'))
    
    # Mark outliers
    df['Is_Outlier_ZScore'] = np.abs(df['Z_Score']) > threshold
    
    return df


def visualize_outliers(df, lower_bound, upper_bound):
    """Visualize outliers using various plots"""
    print("\n" + "=" * 50)
    print("VISUALIZING OUTLIERS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Box plot
    axes[0, 0].boxplot(df['Value'])
    axes[0, 0].set_title('Box Plot')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
    axes[0, 0].axhline(y=upper_bound, color='r', linestyle='--', label='Upper Bound')
    axes[0, 0].legend()
    
    # Histogram
    axes[0, 1].hist(df['Value'], bins=30, edgecolor='black')
    axes[0, 1].axvline(x=lower_bound, color='r', linestyle='--', label='Lower Bound')
    axes[0, 1].axvline(x=upper_bound, color='r', linestyle='--', label='Upper Bound')
    axes[0, 1].set_title('Histogram')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Scatter plot with index
    axes[1, 0].scatter(df.index, df['Value'], c=df['Is_Outlier_IQR'].map({True: 'red', False: 'blue'}), alpha=0.6)
    axes[1, 0].axhline(y=lower_bound, color='r', linestyle='--')
    axes[1, 0].axhline(y=upper_bound, color='r', linestyle='--')
    axes[1, 0].set_title('Scatter Plot (IQR Method)')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Value')
    
    # Z-score plot
    axes[1, 1].scatter(df.index, df['Z_Score'], c=df['Is_Outlier_ZScore'].map({True: 'red', False: 'blue'}), alpha=0.6)
    axes[1, 1].axhline(y=3, color='r', linestyle='--', label='Threshold (+3)')
    axes[1, 1].axhline(y=-3, color='r', linestyle='--', label='Threshold (-3)')
    axes[1, 1].set_title('Z-Score Plot')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel('Z-Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'outlier_visualization.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def treat_outliers(df, lower_bound, upper_bound):
    """Demonstrate different outlier treatment strategies"""
    print("\n" + "=" * 50)
    print("OUTLIER TREATMENT STRATEGIES")
    print("=" * 50)
    
    print(f"\nOriginal data statistics:")
    print(df['Value'].describe())
    
    # Strategy 1: Remove outliers
    print("\n1. Remove outliers:")
    df_removed = df[~df['Is_Outlier_IQR']].copy()
    print(f"Rows before: {len(df)}")
    print(f"Rows after: {len(df_removed)}")
    print(f"Removed: {len(df) - len(df_removed)}")
    print(f"\nStatistics after removal:")
    print(df_removed['Value'].describe())
    
    # Strategy 2: Cap outliers (Winsorization)
    print("\n2. Cap outliers (Winsorization):")
    df_capped = df.copy()
    df_capped['Value_Capped'] = df_capped['Value'].clip(lower=lower_bound, upper=upper_bound)
    print(f"Values capped to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"\nStatistics after capping:")
    print(df_capped['Value_Capped'].describe())
    
    # Strategy 3: Replace with median
    print("\n3. Replace outliers with median:")
    df_replaced = df.copy()
    median_value = df_replaced['Value'].median()
    df_replaced['Value_Replaced'] = df_replaced.apply(
        lambda row: median_value if row['Is_Outlier_IQR'] else row['Value'], 
        axis=1
    )
    print(f"Outliers replaced with median: {median_value:.2f}")
    print(f"\nStatistics after replacement:")
    print(df_replaced['Value_Replaced'].describe())
    
    # Strategy 4: Log transformation
    print("\n4. Log transformation (for positive skewed data):")
    # Add constant to handle any negative values
    df_log = df.copy()
    min_val = df_log['Value'].min()
    if min_val <= 0:
        df_log['Value_Log'] = np.log1p(df_log['Value'] - min_val + 1)
    else:
        df_log['Value_Log'] = np.log(df_log['Value'])
    print(f"\nStatistics after log transformation:")
    print(df_log['Value_Log'].describe())
    
    return df_removed, df_capped, df_replaced


def compare_methods():
    """Compare IQR and Z-score methods"""
    print("\n" + "=" * 50)
    print("COMPARING OUTLIER DETECTION METHODS")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    
    # Apply both methods
    df, lower_bound, upper_bound = iqr_method(df)
    df = z_score_method(df)
    
    # Compare results
    print("\nComparison of methods:")
    print(f"Outliers detected by IQR: {df['Is_Outlier_IQR'].sum()}")
    print(f"Outliers detected by Z-Score: {df['Is_Outlier_ZScore'].sum()}")
    
    # Find agreement
    both_methods = df['Is_Outlier_IQR'] & df['Is_Outlier_ZScore']
    print(f"Outliers detected by both methods: {both_methods.sum()}")
    
    # Find disagreement
    only_iqr = df['Is_Outlier_IQR'] & ~df['Is_Outlier_ZScore']
    only_zscore = ~df['Is_Outlier_IQR'] & df['Is_Outlier_ZScore']
    print(f"Outliers detected only by IQR: {only_iqr.sum()}")
    print(f"Outliers detected only by Z-Score: {only_zscore.sum()}")


def main():
    """Main function to run all examples"""
    print("Welcome to Outlier Detection and Treatment!\n")
    
    # Create sample data
    df = create_sample_data()
    
    # IQR method
    df, lower_bound, upper_bound = iqr_method(df)
    
    # Z-score method
    df = z_score_method(df)
    
    # Visualize
    visualize_outliers(df, lower_bound, upper_bound)
    
    # Treat outliers
    treat_outliers(df, lower_bound, upper_bound)
    
    # Compare methods
    compare_methods()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
