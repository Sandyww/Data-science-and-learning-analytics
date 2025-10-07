"""
CSV Operations with Pandas

This script demonstrates:
- Reading CSV files
- Writing CSV files
- Handling different delimiters
- Working with headers
- Data type specifications
"""

import pandas as pd
import numpy as np
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')

# Create datasets directory if it doesn't exist
os.makedirs(DATASETS_DIR, exist_ok=True)


def create_sample_csv():
    """Create sample CSV files for demonstration"""
    print("=" * 50)
    print("CREATING SAMPLE CSV FILES")
    print("=" * 50)
    
    # Create sample employee data
    employee_data = {
        'ID': [101, 102, 103, 104, 105],
        'Name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'Department': ['IT', 'HR', 'Finance', 'IT', 'Marketing'],
        'Salary': [75000, 55000, 68000, 82000, 62000],
        'Hire_Date': ['2020-01-15', '2019-03-20', '2021-06-10', '2018-11-05', '2022-02-28']
    }
    df = pd.DataFrame(employee_data)
    
    # Save to CSV
    csv_path = os.path.join(DATASETS_DIR, 'employees.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n1. Created employees.csv at {csv_path}")
    print(df)
    
    # Create sample sales data
    sales_data = {
        'Date': pd.date_range('2024-01-01', periods=10),
        'Product': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'Quantity': [5, 7, 3, 8, 6, 4, 9, 5, 7, 6],
        'Price': [100.50, 150.75, 89.99, 100.50, 150.75, 89.99, 100.50, 150.75, 89.99, 100.50]
    }
    df_sales = pd.DataFrame(sales_data)
    
    # Save to CSV with different delimiter
    tsv_path = os.path.join(DATASETS_DIR, 'sales.tsv')
    df_sales.to_csv(tsv_path, index=False, sep='\t')
    print(f"\n2. Created sales.tsv (tab-separated) at {tsv_path}")
    print(df_sales.head())
    
    return csv_path, tsv_path


def reading_csv():
    """Demonstrate reading CSV files"""
    print("\n" + "=" * 50)
    print("READING CSV FILES")
    print("=" * 50)
    
    csv_path = os.path.join(DATASETS_DIR, 'employees.csv')
    
    # Basic CSV reading
    print("\n1. Basic CSV reading:")
    df = pd.read_csv(csv_path)
    print(df)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Reading with specific columns
    print("\n2. Reading specific columns:")
    df_subset = pd.read_csv(csv_path, usecols=['Name', 'Department', 'Salary'])
    print(df_subset)
    
    # Reading with custom index
    print("\n3. Reading with custom index column:")
    df_indexed = pd.read_csv(csv_path, index_col='ID')
    print(df_indexed)
    
    # Reading with data type specification
    print("\n4. Reading with data type specification:")
    dtype_dict = {
        'ID': int,
        'Name': str,
        'Department': str,
        'Salary': float
    }
    df_typed = pd.read_csv(csv_path, dtype=dtype_dict)
    print(df_typed.dtypes)
    
    # Reading tab-separated file
    print("\n5. Reading tab-separated file:")
    tsv_path = os.path.join(DATASETS_DIR, 'sales.tsv')
    df_tsv = pd.read_csv(tsv_path, sep='\t')
    print(df_tsv.head())


def writing_csv():
    """Demonstrate writing CSV files"""
    print("\n" + "=" * 50)
    print("WRITING CSV FILES")
    print("=" * 50)
    
    # Create sample data
    data = {
        'Student': ['John', 'Emma', 'Michael', 'Sophia', 'William'],
        'Math': [85, 92, 78, 95, 88],
        'Science': [90, 87, 92, 89, 94],
        'English': [88, 91, 85, 93, 86]
    }
    df = pd.DataFrame(data)
    print("\nStudent Grades:")
    print(df)
    
    # Write to CSV
    output_path = os.path.join(DATASETS_DIR, 'student_grades.csv')
    df.to_csv(output_path, index=False)
    print(f"\n1. Saved to {output_path}")
    
    # Write with index
    output_path_indexed = os.path.join(DATASETS_DIR, 'student_grades_indexed.csv')
    df.to_csv(output_path_indexed, index=True)
    print(f"2. Saved with index to {output_path_indexed}")
    
    # Write specific columns
    output_path_subset = os.path.join(DATASETS_DIR, 'student_math_science.csv')
    df[['Student', 'Math', 'Science']].to_csv(output_path_subset, index=False)
    print(f"3. Saved subset to {output_path_subset}")
    
    # Write with custom separator
    output_path_pipe = os.path.join(DATASETS_DIR, 'student_grades_pipe.txt')
    df.to_csv(output_path_pipe, index=False, sep='|')
    print(f"4. Saved with pipe separator to {output_path_pipe}")


def advanced_csv_operations():
    """Demonstrate advanced CSV operations"""
    print("\n" + "=" * 50)
    print("ADVANCED CSV OPERATIONS")
    print("=" * 50)
    
    # Create data with missing values
    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, np.nan, 35, 28, np.nan],
        'City': ['NYC', 'LA', None, 'Chicago', 'Boston'],
        'Score': [85.5, 92.0, 78.5, np.nan, 88.0]
    }
    df = pd.DataFrame(data)
    print("\nData with missing values:")
    print(df)
    
    # Save with missing values
    csv_path = os.path.join(DATASETS_DIR, 'data_with_missing.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n1. Saved data with missing values to {csv_path}")
    
    # Read and handle missing values
    print("\n2. Reading and identifying missing values:")
    df_read = pd.read_csv(csv_path)
    print(df_read)
    print("\nMissing values per column:")
    print(df_read.isnull().sum())
    
    # Append to existing CSV
    print("\n3. Appending to existing CSV:")
    new_data = pd.DataFrame({
        'Name': ['Frank', 'Grace'],
        'Age': [30, 27],
        'City': ['Seattle', 'Austin'],
        'Score': [89.5, 91.0]
    })
    new_data.to_csv(csv_path, mode='a', header=False, index=False)
    print("Appended new data. Reading updated file:")
    df_updated = pd.read_csv(csv_path)
    print(df_updated)
    
    # Reading large files in chunks
    print("\n4. Reading CSV in chunks (demonstration):")
    chunk_size = 2
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
        print(f"\nChunk {i + 1}:")
        print(chunk)


def main():
    """Main function to run all examples"""
    print("Welcome to CSV Operations with Pandas!\n")
    
    # Create sample files
    create_sample_csv()
    
    # Demonstrate reading
    reading_csv()
    
    # Demonstrate writing
    writing_csv()
    
    # Demonstrate advanced operations
    advanced_csv_operations()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print(f"Sample files saved in: {DATASETS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
