import pandas as pd
import argparse

# Function to print category counts for each feature in a dataset
def print_category_counts(file_path):
    df = pd.read_csv(file_path)
    
    result = {}
    for column in df.columns:
        if column not in ['path', 'label']:
            category_counts = df[column].value_counts().to_dict()
            result[column] = category_counts
    
    for column, counts in result.items():
        print(f'Column: {column}')
        for category, count in counts.items():
            print(f'    {category}: {count}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print category counts for each feature in a dataset.")
    parser.add_argument('--file_path', required=True, help="CSV file path.")
    args = parser.parse_args()
    
    print_category_counts(args.file_path)
