import pandas as pd
import argparse

# Apply weighting to hard examples (e.g., shadowed images)
def weight_hard_examples(input_csv, output_csv, column='shadow', value=1, multiplier=4):
    df = pd.read_csv(input_csv)
    filtered_df = df[df[column] == value]
    duplicated_df = pd.concat([filtered_df] * multiplier, ignore_index=True)
    weighted_df = pd.concat([df, duplicated_df], ignore_index=True)
    weighted_df.to_csv(output_csv, index=False)
    print(f"Weighted dataset saved to {output_csv}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply weighting to hard examples.")
    parser.add_argument('--input_csv', required=True, help="Input CSV file path.")
    parser.add_argument('--output_csv', required=True, help="Output CSV file path.")
    parser.add_argument('--column', default='shadow', help="Column name for hard example identification.")
    parser.add_argument('--value', type=int, default=1, help="Value indicating hard examples.")
    parser.add_argument('--multiplier', type=int, default=4, help="Multiplier for hard examples.")
    args = parser.parse_args()
    
    weight_hard_examples(args.input_csv, args.output_csv, args.column, args.value, args.multiplier)
