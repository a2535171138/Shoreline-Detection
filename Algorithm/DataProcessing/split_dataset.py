import pandas as pd
import argparse

# Split dataset into training and test sets
def split_dataset(input_csv, train_csv, test_csv, num_train, num_test):
    df = pd.read_csv(input_csv)
    shuffled_df = df.sample(frac=1)
    train_df = shuffled_df.iloc[:num_train]
    test_df = shuffled_df.iloc[num_train:(num_train + num_test)]
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Training set saved to {train_csv}")
    print(f"Test set saved to {test_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and test sets.")
    parser.add_argument('--input_csv', required=True, help="Input CSV file path.")
    parser.add_argument('--train_csv', required=True, help="Output CSV file path for training set.")
    parser.add_argument('--test_csv', required=True, help="Output CSV file path for test set.")
    parser.add_argument('--num_train', type=int, required=True, help="Number of training samples.")
    parser.add_argument('--num_test', type=int, required=True, help="Number of test samples.")
    args = parser.parse_args()
    
    split_dataset(args.input_csv, args.train_csv, args.test_csv, args.num_train, args.num_test)
