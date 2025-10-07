"""
Data Manipulation with Pandas

This script demonstrates common data manipulation operations:
- Merging and joining DataFrames
- Reshaping data (pivot, melt)
- Handling duplicates
- Renaming columns
"""

import pandas as pd
import numpy as np

def merging_dataframes():
    """Demonstrate merging and joining operations"""
    print("=" * 50)
    print("MERGING AND JOINING DATAFRAMES")
    print("=" * 50)
    
    # Create sample DataFrames
    df1 = pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Department': ['HR', 'IT', 'Finance', 'IT']
    })
    
    df2 = pd.DataFrame({
        'ID': [1, 2, 5, 6],
        'Salary': [50000, 60000, 55000, 65000],
        'Years': [3, 5, 2, 7]
    })
    
    print("\nDataFrame 1 (Employee Info):")
    print(df1)
    print("\nDataFrame 2 (Salary Info):")
    print(df2)
    
    # Inner join
    print("\n1. Inner Join (matching records only):")
    inner_merge = pd.merge(df1, df2, on='ID', how='inner')
    print(inner_merge)
    
    # Left join
    print("\n2. Left Join (all records from left):")
    left_merge = pd.merge(df1, df2, on='ID', how='left')
    print(left_merge)
    
    # Outer join
    print("\n3. Outer Join (all records from both):")
    outer_merge = pd.merge(df1, df2, on='ID', how='outer')
    print(outer_merge)


def reshaping_data():
    """Demonstrate data reshaping operations"""
    print("\n" + "=" * 50)
    print("RESHAPING DATA")
    print("=" * 50)
    
    # Create sample data
    data = {
        'Date': ['2024-01', '2024-01', '2024-02', '2024-02'],
        'City': ['NYC', 'LA', 'NYC', 'LA'],
        'Temperature': [32, 65, 45, 70],
        'Humidity': [60, 40, 55, 35]
    }
    df = pd.DataFrame(data)
    print("\nOriginal Data:")
    print(df)
    
    # Pivot table
    print("\n1. Pivot Table (Temperature by City and Date):")
    pivot = df.pivot(index='Date', columns='City', values='Temperature')
    print(pivot)
    
    # Melt (unpivot)
    print("\n2. Melting data (long format):")
    melted = df.melt(id_vars=['Date', 'City'], 
                     value_vars=['Temperature', 'Humidity'],
                     var_name='Metric', 
                     value_name='Value')
    print(melted)


def handling_duplicates():
    """Demonstrate handling duplicate data"""
    print("\n" + "=" * 50)
    print("HANDLING DUPLICATES")
    print("=" * 50)
    
    # Create DataFrame with duplicates
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob'],
        'Age': [25, 30, 35, 25, 30],
        'City': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
    }
    df = pd.DataFrame(data)
    print("\nOriginal Data (with duplicates):")
    print(df)
    
    # Check for duplicates
    print("\n1. Check for duplicate rows:")
    print(df.duplicated())
    
    # Remove duplicates
    print("\n2. Remove duplicate rows:")
    df_no_dup = df.drop_duplicates()
    print(df_no_dup)
    
    # Remove duplicates based on specific columns
    print("\n3. Remove duplicates based on 'Name' column:")
    df_no_dup_name = df.drop_duplicates(subset=['Name'])
    print(df_no_dup_name)


def renaming_operations():
    """Demonstrate renaming operations"""
    print("\n" + "=" * 50)
    print("RENAMING OPERATIONS")
    print("=" * 50)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9]
    })
    print("\nOriginal DataFrame:")
    print(df)
    
    # Rename columns
    print("\n1. Rename specific columns:")
    df_renamed = df.rename(columns={'col1': 'A', 'col2': 'B'})
    print(df_renamed)
    
    # Rename all columns
    print("\n2. Rename all columns:")
    df.columns = ['X', 'Y', 'Z']
    print(df)
    
    # Rename using function
    print("\n3. Rename using function (uppercase):")
    df.columns = df.columns.str.upper()
    print(df)


def data_concatenation():
    """Demonstrate concatenating DataFrames"""
    print("\n" + "=" * 50)
    print("CONCATENATING DATAFRAMES")
    print("=" * 50)
    
    # Create sample DataFrames
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    df2 = pd.DataFrame({
        'A': [7, 8, 9],
        'B': [10, 11, 12]
    })
    
    df3 = pd.DataFrame({
        'C': [13, 14, 15],
        'D': [16, 17, 18]
    })
    
    print("\nDataFrame 1:")
    print(df1)
    print("\nDataFrame 2:")
    print(df2)
    print("\nDataFrame 3:")
    print(df3)
    
    # Vertical concatenation
    print("\n1. Vertical concatenation (rows):")
    vertical_concat = pd.concat([df1, df2], ignore_index=True)
    print(vertical_concat)
    
    # Horizontal concatenation
    print("\n2. Horizontal concatenation (columns):")
    horizontal_concat = pd.concat([df1, df3], axis=1)
    print(horizontal_concat)


def main():
    """Main function to run all examples"""
    print("Welcome to Data Manipulation with Pandas!\n")
    
    # Run all demonstrations
    merging_dataframes()
    reshaping_data()
    handling_duplicates()
    renaming_operations()
    data_concatenation()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
