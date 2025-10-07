"""
Grouping and Aggregation with Pandas

This script demonstrates:
- GroupBy operations
- Aggregation functions
- Multiple aggregations
- Transform operations
"""

import pandas as pd
import numpy as np

def basic_groupby():
    """Demonstrate basic groupby operations"""
    print("=" * 50)
    print("BASIC GROUPBY OPERATIONS")
    print("=" * 50)
    
    # Create sample sales data
    data = {
        'Product': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B'],
        'Region': ['East', 'East', 'West', 'West', 'East', 'West', 'East', 'East'],
        'Sales': [100, 150, 120, 180, 110, 90, 95, 160],
        'Quantity': [5, 7, 6, 9, 5, 4, 4, 8]
    }
    df = pd.DataFrame(data)
    print("\nSales Data:")
    print(df)
    
    # Group by single column
    print("\n1. Group by Product and sum Sales:")
    product_sales = df.groupby('Product')['Sales'].sum()
    print(product_sales)
    
    # Group by multiple columns
    print("\n2. Group by Product and Region:")
    product_region_sales = df.groupby(['Product', 'Region'])['Sales'].sum()
    print(product_region_sales)
    
    # Multiple aggregations
    print("\n3. Multiple aggregation functions:")
    stats = df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count', 'min', 'max'])
    print(stats)


def advanced_aggregation():
    """Demonstrate advanced aggregation techniques"""
    print("\n" + "=" * 50)
    print("ADVANCED AGGREGATION")
    print("=" * 50)
    
    # Create employee data
    data = {
        'Department': ['IT', 'IT', 'HR', 'HR', 'Finance', 'Finance', 'IT'],
        'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
        'Salary': [70000, 75000, 50000, 52000, 65000, 68000, 72000],
        'Experience': [5, 7, 3, 4, 6, 8, 5]
    }
    df = pd.DataFrame(data)
    print("\nEmployee Data:")
    print(df)
    
    # Aggregate different columns differently
    print("\n1. Different aggregations for different columns:")
    agg_dict = {
        'Salary': ['mean', 'sum'],
        'Experience': ['mean', 'max']
    }
    dept_stats = df.groupby('Department').agg(agg_dict)
    print(dept_stats)
    
    # Custom aggregation function
    print("\n2. Custom aggregation (salary range):")
    def salary_range(x):
        return x.max() - x.min()
    
    dept_range = df.groupby('Department')['Salary'].agg(salary_range)
    print(dept_range)
    
    # Named aggregations
    print("\n3. Named aggregations:")
    dept_summary = df.groupby('Department').agg(
        avg_salary=('Salary', 'mean'),
        total_employees=('Employee', 'count'),
        max_experience=('Experience', 'max')
    )
    print(dept_summary)


def transform_operations():
    """Demonstrate transform operations"""
    print("\n" + "=" * 50)
    print("TRANSFORM OPERATIONS")
    print("=" * 50)
    
    # Create sample data
    data = {
        'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Value': [10, 20, 30, 40, 50, 60]
    }
    df = pd.DataFrame(data)
    print("\nOriginal Data:")
    print(df)
    
    # Add group mean as new column
    print("\n1. Add group mean as new column:")
    df['Group_Mean'] = df.groupby('Category')['Value'].transform('mean')
    print(df)
    
    # Normalize within groups
    print("\n2. Normalize values within groups:")
    df['Normalized'] = df.groupby('Category')['Value'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print(df)
    
    # Fill missing values with group mean
    print("\n3. Fill missing values with group mean:")
    data_with_nan = {
        'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
        'Value': [10, np.nan, 30, np.nan, 50, 60]
    }
    df_nan = pd.DataFrame(data_with_nan)
    print("\nData with NaN:")
    print(df_nan)
    
    df_nan['Value_Filled'] = df_nan.groupby('Category')['Value'].transform(
        lambda x: x.fillna(x.mean())
    )
    print("\nAfter filling with group mean:")
    print(df_nan)


def pivot_tables():
    """Demonstrate pivot table operations"""
    print("\n" + "=" * 50)
    print("PIVOT TABLES")
    print("=" * 50)
    
    # Create sample sales data
    data = {
        'Date': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-01', '2024-02'],
        'Product': ['A', 'B', 'A', 'B', 'C', 'C'],
        'Region': ['East', 'East', 'East', 'East', 'West', 'West'],
        'Sales': [100, 150, 120, 180, 90, 95]
    }
    df = pd.DataFrame(data)
    print("\nSales Data:")
    print(df)
    
    # Simple pivot table
    print("\n1. Pivot table (Sales by Date and Product):")
    pivot1 = pd.pivot_table(df, values='Sales', index='Date', columns='Product', aggfunc='sum')
    print(pivot1)
    
    # Pivot table with multiple aggregations
    print("\n2. Pivot table with mean and sum:")
    pivot2 = pd.pivot_table(df, values='Sales', index='Date', columns='Product', 
                            aggfunc=['mean', 'sum'], fill_value=0)
    print(pivot2)
    
    # Pivot table with margins
    print("\n3. Pivot table with margins (totals):")
    pivot3 = pd.pivot_table(df, values='Sales', index='Date', columns='Product', 
                            aggfunc='sum', fill_value=0, margins=True)
    print(pivot3)


def main():
    """Main function to run all examples"""
    print("Welcome to Grouping and Aggregation with Pandas!\n")
    
    # Run all demonstrations
    basic_groupby()
    advanced_aggregation()
    transform_operations()
    pivot_tables()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
