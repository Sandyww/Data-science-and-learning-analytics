"""
Pandas Basics - Introduction to Series and DataFrames

This script demonstrates the fundamental concepts of Pandas:
- Creating Series and DataFrames
- Basic properties and methods
- Data types and indexing
"""

import pandas as pd
import numpy as np

def series_examples():
    """Demonstrate Pandas Series operations"""
    print("=" * 50)
    print("PANDAS SERIES EXAMPLES")
    print("=" * 50)
    
    # Creating a Series from a list
    print("\n1. Creating a Series from a list:")
    data = [10, 20, 30, 40, 50]
    series1 = pd.Series(data)
    print(series1)
    
    # Creating a Series with custom index
    print("\n2. Creating a Series with custom index:")
    series2 = pd.Series(data, index=['a', 'b', 'c', 'd', 'e'])
    print(series2)
    
    # Creating a Series from a dictionary
    print("\n3. Creating a Series from a dictionary:")
    data_dict = {'apple': 5, 'banana': 3, 'orange': 8, 'grape': 12}
    series3 = pd.Series(data_dict)
    print(series3)
    
    # Accessing Series elements
    print("\n4. Accessing Series elements:")
    print(f"Element at index 'apple': {series3['apple']}")
    print(f"First three elements:\n{series3[:3]}")
    
    # Basic Series operations
    print("\n5. Basic Series operations:")
    print(f"Sum: {series3.sum()}")
    print(f"Mean: {series3.mean()}")
    print(f"Max: {series3.max()}")
    print(f"Min: {series3.min()}")


def dataframe_examples():
    """Demonstrate Pandas DataFrame operations"""
    print("\n" + "=" * 50)
    print("PANDAS DATAFRAME EXAMPLES")
    print("=" * 50)
    
    # Creating a DataFrame from a dictionary
    print("\n1. Creating a DataFrame from a dictionary:")
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 32],
        'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
        'Score': [85, 92, 78, 95, 88]
    }
    df = pd.DataFrame(data)
    print(df)
    
    # DataFrame properties
    print("\n2. DataFrame properties:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index: {df.index.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    
    # Viewing data
    print("\n3. Viewing data:")
    print("First 3 rows:")
    print(df.head(3))
    print("\nLast 2 rows:")
    print(df.tail(2))
    
    # Basic statistics
    print("\n4. Basic statistics:")
    print(df.describe())
    
    # Selecting columns
    print("\n5. Selecting columns:")
    print("Names column:")
    print(df['Name'])
    print("\nMultiple columns:")
    print(df[['Name', 'Score']])
    
    # Selecting rows
    print("\n6. Selecting rows by index:")
    print(df.iloc[0])  # First row
    print("\nRows 1 to 3:")
    print(df.iloc[1:4])
    
    # Filtering data
    print("\n7. Filtering data:")
    print("People with Score > 85:")
    high_scorers = df[df['Score'] > 85]
    print(high_scorers)
    
    # Adding a new column
    print("\n8. Adding a new column:")
    df['Grade'] = df['Score'].apply(lambda x: 'A' if x >= 90 else ('B' if x >= 80 else 'C'))
    print(df)
    
    return df


def advanced_operations(df):
    """Demonstrate advanced DataFrame operations"""
    print("\n" + "=" * 50)
    print("ADVANCED DATAFRAME OPERATIONS")
    print("=" * 50)
    
    # Sorting
    print("\n1. Sorting by Score (descending):")
    df_sorted = df.sort_values('Score', ascending=False)
    print(df_sorted)
    
    # Modifying values
    print("\n2. Modifying values:")
    df_copy = df.copy()
    df_copy.loc[0, 'Score'] = 90
    print(df_copy)
    
    # Creating new DataFrame with random data
    print("\n3. Working with numerical data:")
    np.random.seed(42)
    numerical_df = pd.DataFrame({
        'A': np.random.randn(10),
        'B': np.random.randn(10),
        'C': np.random.randn(10)
    })
    print(numerical_df.head())
    
    # Column-wise operations
    print("\n4. Column-wise operations:")
    print(f"Mean of each column:\n{numerical_df.mean()}")
    print(f"\nSum of each column:\n{numerical_df.sum()}")
    
    # Row-wise operations
    print("\n5. Row-wise sum:")
    numerical_df['Row_Sum'] = numerical_df.sum(axis=1)
    print(numerical_df.head())


def main():
    """Main function to run all examples"""
    print("Welcome to Pandas Basics!\n")
    
    # Run Series examples
    series_examples()
    
    # Run DataFrame examples
    df = dataframe_examples()
    
    # Run advanced operations
    advanced_operations(df)
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
