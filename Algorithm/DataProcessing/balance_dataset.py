import pandas as pd
import argparse

# Balance dataset by upsampling
def balance_dataset(input_csv, output_csv, column):
    df = pd.read_csv(input_csv)
    column_counts = df[column].value_counts().to_dict()
    max_count = max(column_counts.values())

    samples = []
    for value, count in column_counts.items():
        value_df = df[df[column] == value]
        if count < max_count:
            sampled_value_df = value_df.sample(max_count, replace=True, random_state=42)
            samples.append(sampled_value_df)
        else:
            samples.append(value_df)
    
    balanced_df = pd.concat(samples).reset_index(drop=True)
    balanced_df.to_csv(output_csv, index=False)
    print(f"Balanced dataset saved to {output_csv}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Balance dataset by upsampling.")
    parser.add_argument('--input_csv', required=True, help="Input CSV file path.")
    parser.add_argument('--output_csv', required=True, help="Output CSV file path.")
    parser.add_argument('--column', required=True, help="Column name to balance by.")
    args = parser.parse_args()
    
    balance_dataset(args.input_csv, args.output_csv, args.column)
