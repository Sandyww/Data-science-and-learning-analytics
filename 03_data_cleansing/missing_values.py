"""
Handling Missing Values

This script demonstrates:
- Detecting missing values
- Different strategies for handling missing data
- Imputation techniques
- Forward and backward filling
"""

import pandas as pd
import numpy as np

def detect_missing_values():
    """Demonstrate detection of missing values"""
    print("=" * 50)
    print("DETECTING MISSING VALUES")
    print("=" * 50)
    
    # Create DataFrame with missing values
    data = {
        'Name': ['Alice', 'Bob', np.nan, 'David', 'Eve'],
        'Age': [25, np.nan, 35, 28, np.nan],
        'City': ['NYC', 'LA', 'Chicago', None, 'Boston'],
        'Salary': [75000, 65000, np.nan, 80000, 70000],
        'Department': ['IT', 'HR', 'Finance', 'IT', None]
    }
    df = pd.DataFrame(data)
    
    print("\nDataFrame with missing values:")
    print(df)
    
    # Check for missing values
    print("\n1. Check for missing values (boolean mask):")
    print(df.isnull())
    
    print("\n2. Count missing values per column:")
    print(df.isnull().sum())
    
    print("\n3. Percentage of missing values per column:")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct)
    
    print("\n4. Total missing values in DataFrame:")
    print(f"Total missing: {df.isnull().sum().sum()}")
    
    print("\n5. Rows with any missing values:")
    print(df[df.isnull().any(axis=1)])
    
    return df


def dropping_missing_values(df):
    """Demonstrate dropping missing values"""
    print("\n" + "=" * 50)
    print("DROPPING MISSING VALUES")
    print("=" * 50)
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Drop rows with any missing values
    print("\n1. Drop rows with any missing values:")
    df_dropped_rows = df.dropna()
    print(df_dropped_rows)
    print(f"Rows remaining: {len(df_dropped_rows)}/{len(df)}")
    
    # Drop columns with any missing values
    print("\n2. Drop columns with any missing values:")
    df_dropped_cols = df.dropna(axis=1)
    print(df_dropped_cols)
    print(f"Columns remaining: {len(df_dropped_cols.columns)}/{len(df.columns)}")
    
    # Drop rows with all missing values
    data_with_empty = df.copy()
    data_with_empty.loc[5] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    print("\n3. DataFrame with completely empty row:")
    print(data_with_empty)
    
    df_dropped_all = data_with_empty.dropna(how='all')
    print("\nAfter dropping rows with all missing values:")
    print(df_dropped_all)
    
    # Drop rows with missing values in specific columns
    print("\n4. Drop rows with missing values in specific columns:")
    df_dropped_subset = df.dropna(subset=['Age', 'Salary'])
    print(df_dropped_subset)


def filling_missing_values(df):
    """Demonstrate filling missing values"""
    print("\n" + "=" * 50)
    print("FILLING MISSING VALUES")
    print("=" * 50)
    
    print("\nOriginal DataFrame:")
    print(df)
    
    # Fill with a constant value
    print("\n1. Fill all missing values with a constant:")
    df_filled_constant = df.fillna(0)
    print(df_filled_constant)
    
    # Fill with different values for different columns
    print("\n2. Fill with different values per column:")
    fill_values = {
        'Name': 'Unknown',
        'Age': df['Age'].mean(),
        'City': 'Not Specified',
        'Salary': df['Salary'].median(),
        'Department': 'Unassigned'
    }
    df_filled_dict = df.fillna(fill_values)
    print(df_filled_dict)
    
    # Forward fill
    print("\n3. Forward fill (propagate last valid value):")
    df_ffill = df.fillna(method='ffill')
    print(df_ffill)
    
    # Backward fill
    print("\n4. Backward fill (propagate next valid value):")
    df_bfill = df.fillna(method='bfill')
    print(df_bfill)


def imputation_techniques():
    """Demonstrate advanced imputation techniques"""
    print("\n" + "=" * 50)
    print("ADVANCED IMPUTATION TECHNIQUES")
    print("=" * 50)
    
    # Create numerical data with missing values
    np.random.seed(42)
    data = {
        'Feature1': [1, 2, np.nan, 4, 5, 6, np.nan, 8, 9, 10],
        'Feature2': [10, np.nan, 30, 40, 50, np.nan, 70, 80, 90, 100],
        'Feature3': [100, 200, 300, np.nan, 500, 600, 700, np.nan, 900, 1000]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data with missing values:")
    print(df)
    
    # Mean imputation
    print("\n1. Mean imputation:")
    df_mean = df.fillna(df.mean())
    print(df_mean)
    
    # Median imputation
    print("\n2. Median imputation:")
    df_median = df.fillna(df.median())
    print(df_median)
    
    # Mode imputation (useful for categorical data)
    print("\n3. Mode imputation:")
    categorical_data = {
        'Category': ['A', 'B', np.nan, 'A', 'B', np.nan, 'A', 'C'],
        'Value': [1, 2, 3, 4, 5, 6, 7, 8]
    }
    df_cat = pd.DataFrame(categorical_data)
    print("Original categorical data:")
    print(df_cat)
    
    # Fill with mode
    df_cat['Category'] = df_cat['Category'].fillna(df_cat['Category'].mode()[0])
    print("\nAfter mode imputation:")
    print(df_cat)
    
    # Interpolation
    print("\n4. Interpolation (linear):")
    time_series_data = {
        'Time': range(10),
        'Value': [1, 2, np.nan, 4, 5, np.nan, np.nan, 8, 9, 10]
    }
    df_ts = pd.DataFrame(time_series_data)
    print("Original time series:")
    print(df_ts)
    
    df_ts['Value_Interpolated'] = df_ts['Value'].interpolate()
    print("\nAfter interpolation:")
    print(df_ts)


def handling_missing_in_groups():
    """Demonstrate group-wise imputation"""
    print("\n" + "=" * 50)
    print("GROUP-WISE IMPUTATION")
    print("=" * 50)
    
    # Create data with groups
    data = {
        'Department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance'],
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
        'Salary': [75000, np.nan, 55000, np.nan, 68000, 72000]
    }
    df = pd.DataFrame(data)
    
    print("\nData with missing salaries:")
    print(df)
    
    # Fill with group mean
    print("\n1. Fill with department-wise mean:")
    df['Salary_Filled'] = df.groupby('Department')['Salary'].transform(
        lambda x: x.fillna(x.mean())
    )
    print(df)
    
    # Fill with group median
    print("\n2. Fill with department-wise median:")
    df['Salary_Filled_Median'] = df.groupby('Department')['Salary'].transform(
        lambda x: x.fillna(x.median())
    )
    print(df)


def best_practices():
    """Demonstrate best practices for handling missing data"""
    print("\n" + "=" * 50)
    print("BEST PRACTICES")
    print("=" * 50)
    
    # Create sample data
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Feature1': [10, np.nan, 30, 40, np.nan],
        'Feature2': [100, 200, np.nan, 400, 500],
        'Target': [1, 0, 1, np.nan, 0]
    }
    df = pd.DataFrame(data)
    
    print("\nOriginal data:")
    print(df)
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Strategy 1: Remove rows with missing target
    print("\n1. Remove rows with missing target variable:")
    df_clean = df.dropna(subset=['Target'])
    print(df_clean)
    
    # Strategy 2: Impute features
    print("\n2. Impute missing feature values with mean:")
    for col in ['Feature1', 'Feature2']:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    print(df_clean)
    
    print("\n3. Final check for missing values:")
    print(df_clean.isnull().sum())


def main():
    """Main function to run all examples"""
    print("Welcome to Handling Missing Values!\n")
    
    # Detect missing values
    df = detect_missing_values()
    
    # Dropping strategies
    dropping_missing_values(df.copy())
    
    # Filling strategies
    filling_missing_values(df.copy())
    
    # Advanced imputation
    imputation_techniques()
    
    # Group-wise imputation
    handling_missing_in_groups()
    
    # Best practices
    best_practices()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
