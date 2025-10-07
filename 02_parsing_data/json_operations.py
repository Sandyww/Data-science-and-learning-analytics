"""
JSON Operations with Pandas

This script demonstrates:
- Reading JSON files
- Writing JSON files
- Parsing nested JSON
- Converting between DataFrame and JSON
- Working with JSON from APIs
"""

import pandas as pd
import json
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')

# Create datasets directory if it doesn't exist
os.makedirs(DATASETS_DIR, exist_ok=True)


def create_sample_json():
    """Create sample JSON files for demonstration"""
    print("=" * 50)
    print("CREATING SAMPLE JSON FILES")
    print("=" * 50)
    
    # Simple JSON structure
    simple_data = [
        {"id": 1, "name": "Alice", "age": 25, "city": "NYC"},
        {"id": 2, "name": "Bob", "age": 30, "city": "LA"},
        {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago"}
    ]
    
    simple_json_path = os.path.join(DATASETS_DIR, 'simple_data.json')
    with open(simple_json_path, 'w') as f:
        json.dump(simple_data, f, indent=2)
    print(f"\n1. Created simple_data.json at {simple_json_path}")
    print(json.dumps(simple_data, indent=2))
    
    # Nested JSON structure
    nested_data = {
        "users": [
            {
                "id": 1,
                "name": "Alice",
                "contact": {
                    "email": "alice@example.com",
                    "phone": "123-456-7890"
                },
                "skills": ["Python", "SQL", "Machine Learning"]
            },
            {
                "id": 2,
                "name": "Bob",
                "contact": {
                    "email": "bob@example.com",
                    "phone": "234-567-8901"
                },
                "skills": ["Java", "JavaScript", "React"]
            }
        ]
    }
    
    nested_json_path = os.path.join(DATASETS_DIR, 'nested_data.json')
    with open(nested_json_path, 'w') as f:
        json.dump(nested_data, f, indent=2)
    print(f"\n2. Created nested_data.json at {nested_json_path}")
    print(json.dumps(nested_data, indent=2)[:200] + "...")
    
    return simple_json_path, nested_json_path


def reading_json():
    """Demonstrate reading JSON files"""
    print("\n" + "=" * 50)
    print("READING JSON FILES")
    print("=" * 50)
    
    simple_json_path = os.path.join(DATASETS_DIR, 'simple_data.json')
    
    # Read simple JSON
    print("\n1. Reading simple JSON:")
    df = pd.read_json(simple_json_path)
    print(df)
    print(f"\nDataFrame shape: {df.shape}")
    
    # Read JSON with orient parameter
    print("\n2. Reading JSON with different orientations:")
    
    # Create records-oriented JSON
    df_sample = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    # Save in different formats
    records_path = os.path.join(DATASETS_DIR, 'records_orient.json')
    df_sample.to_json(records_path, orient='records', indent=2)
    
    columns_path = os.path.join(DATASETS_DIR, 'columns_orient.json')
    df_sample.to_json(columns_path, orient='columns', indent=2)
    
    print("Records orientation:")
    print(pd.read_json(records_path))
    
    print("\nColumns orientation:")
    print(pd.read_json(columns_path))
    
    # Read using json module then create DataFrame
    print("\n3. Reading with json module:")
    with open(simple_json_path, 'r') as f:
        data = json.load(f)
    df_from_dict = pd.DataFrame(data)
    print(df_from_dict)


def working_with_nested_json():
    """Demonstrate working with nested JSON structures"""
    print("\n" + "=" * 50)
    print("WORKING WITH NESTED JSON")
    print("=" * 50)
    
    nested_json_path = os.path.join(DATASETS_DIR, 'nested_data.json')
    
    # Read nested JSON
    print("\n1. Reading nested JSON:")
    with open(nested_json_path, 'r') as f:
        data = json.load(f)
    
    # Normalize nested JSON
    print("\n2. Normalizing nested JSON with json_normalize:")
    df = pd.json_normalize(data['users'])
    print(df)
    
    # Flatten with custom separator
    print("\n3. Flatten with custom separator:")
    df_custom = pd.json_normalize(
        data['users'],
        sep='_'
    )
    print(df_custom)
    
    # Extract nested array to separate rows
    print("\n4. Exploding nested arrays:")
    df_exploded = df_custom.explode('skills')
    print(df_exploded[['name', 'skills']])


def writing_json():
    """Demonstrate writing JSON files"""
    print("\n" + "=" * 50)
    print("WRITING JSON FILES")
    print("=" * 50)
    
    # Create sample DataFrame
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
        'Price': [1200, 25, 75, 300],
        'Stock': [50, 200, 150, 75],
        'Rating': [4.5, 4.2, 4.7, 4.6]
    }
    df = pd.DataFrame(data)
    print("\nProduct Data:")
    print(df)
    
    # Write in different orientations
    print("\n1. Writing in records orientation:")
    records_path = os.path.join(DATASETS_DIR, 'products_records.json')
    df.to_json(records_path, orient='records', indent=2)
    print(f"Saved to {records_path}")
    with open(records_path, 'r') as f:
        print(f.read())
    
    print("\n2. Writing in columns orientation:")
    columns_path = os.path.join(DATASETS_DIR, 'products_columns.json')
    df.to_json(columns_path, orient='columns', indent=2)
    print(f"Saved to {columns_path}")
    with open(columns_path, 'r') as f:
        print(f.read())
    
    print("\n3. Writing in index orientation:")
    index_path = os.path.join(DATASETS_DIR, 'products_index.json')
    df.to_json(index_path, orient='index', indent=2)
    print(f"Saved to {index_path}")
    with open(index_path, 'r') as f:
        print(f.read())


def json_to_dataframe_conversions():
    """Demonstrate various JSON to DataFrame conversions"""
    print("\n" + "=" * 50)
    print("JSON TO DATAFRAME CONVERSIONS")
    print("=" * 50)
    
    # JSON string to DataFrame
    print("\n1. JSON string to DataFrame:")
    json_string = '''
    [
        {"name": "Alice", "score": 85},
        {"name": "Bob", "score": 92},
        {"name": "Charlie", "score": 78}
    ]
    '''
    df = pd.read_json(json_string)
    print(df)
    
    # Dictionary to DataFrame to JSON
    print("\n2. Dictionary to DataFrame to JSON:")
    data_dict = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data_dict)
    json_output = df.to_json(orient='records')
    print("JSON output:")
    print(json.dumps(json.loads(json_output), indent=2))
    
    # Complex nested structure
    print("\n3. Complex nested structure:")
    complex_data = [
        {
            "id": 1,
            "name": "Product A",
            "details": {
                "category": "Electronics",
                "price": 299.99
            },
            "tags": ["new", "featured"]
        },
        {
            "id": 2,
            "name": "Product B",
            "details": {
                "category": "Clothing",
                "price": 49.99
            },
            "tags": ["sale", "clearance"]
        }
    ]
    
    df_complex = pd.json_normalize(
        complex_data,
        record_path='tags',
        meta=['id', 'name', ['details', 'category'], ['details', 'price']],
        meta_prefix='product_'
    )
    print(df_complex)


def main():
    """Main function to run all examples"""
    print("Welcome to JSON Operations with Pandas!\n")
    
    # Create sample files
    create_sample_json()
    
    # Demonstrate reading
    reading_json()
    
    # Demonstrate nested JSON
    working_with_nested_json()
    
    # Demonstrate writing
    writing_json()
    
    # Demonstrate conversions
    json_to_dataframe_conversions()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print(f"Sample files saved in: {DATASETS_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
