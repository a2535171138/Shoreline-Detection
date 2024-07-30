import pandas as pd
import os
import argparse

# Function to generate a dictionary for a row and add it to the output list
def get_dict(row, path, site, rectified, new_label, num_no_label):
    if rectified == 0:
        label = row['oblique_geometry']
    else:
        label = row['rectified_geometry']
    
    if not isinstance(label, str):
        num_no_label[0] += 1
        return
    
    new_label.append({
        'path': path,
        'rectified': rectified,
        'site': site,
        'camera': row['camera'],
        'type': row['type'],
        'obstruction': row['obstruction'],
        'downward': row['downward'],
        'low': row['low'],
        'shadow': row['shadow'],
        'label': label
    })

# Function to process DataFrames and generate output labels
def process_dataframes(csv_files, folders, output_csv):
    # Reading CSV files into DataFrames
    df1 = pd.read_csv(csv_files[0], index_col=0)
    df2 = pd.read_csv(csv_files[1], index_col=0)
    df3 = pd.read_csv(csv_files[2], index_col=0)
    df4 = pd.read_csv(csv_files[3])
    
    num_existed_files = 0
    num_not_existed_files = 0
    num_no_label = [0]
    output_label = []

    # Processing df1 (CoastSnap)
    for index, row in df1.iterrows():
        oblique_image_path = f"{folders[0]}/{row['site']}/{row['oblique_image']}"
        if os.path.exists(oblique_image_path):
            get_dict(row, oblique_image_path, 'CoastSnap', 0, output_label, num_no_label)
            num_existed_files += 1
        else:
            num_not_existed_files += 1

    # Processing df2 (Goldcoast)
    for index, row in df2.iterrows():
        image_path = f"{folders[1]}/{row['site']}/c{row['camera']}/{row['oblique_image']}"
        if os.path.exists(image_path):
            get_dict(row, image_path, 'Goldcoast', 0, output_label, num_no_label)
            num_existed_files += 1
        else:
            num_not_existed_files += 1

    # Processing df3 (Narrabeen)
    for index, row in df3.iterrows():
        oblique_image_path = f"{folders[2]}/{row['oblique_image']}"
        if os.path.exists(oblique_image_path):
            get_dict(row, oblique_image_path, 'Narrabeen', 0, output_label, num_no_label)
            num_existed_files += 1
        else:
            num_not_existed_files += 1

    # Processing df4 (plan.csv)
    for index, row in df4.iterrows():
        path = row['path']
        label = row['label']
        
        if not isinstance(label, str):
            num_no_label[0] += 1
            continue
        
        if os.path.exists(path):
            output_label.append({
                'path': row['path'],
                'rectified': 1,
                'site': 'Rectified',
                'camera': row['camera'],
                'type': row['type'],
                'obstruction': row['obstruction'],
                'downward': row['downward'],
                'low': row['low'],
                'shadow': row['shadow'],
                'label': row['label']
            })
            num_existed_files += 1
        else:
            num_not_existed_files += 1
    
    output_df = pd.DataFrame(output_label)
    output_df.to_csv(output_csv, index=False)
    
    print('Existed files num:', num_existed_files)
    print('Not existed files num:', num_not_existed_files)
    print('No label num:', num_no_label[0])
    print('Output labels num:', len(output_label))
    print(f"Output labels saved to {output_csv}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process DataFrames and generate output labels.")
    parser.add_argument('--csv_files', nargs='+', required=True, help="List of input CSV file paths.")
    parser.add_argument('--folders', nargs='+', required=True, help="List of folder paths corresponding to the CSV files.")
    parser.add_argument('--output_csv', required=True, help="Output CSV file path.")
    args = parser.parse_args()
    
    process_dataframes(args.csv_files, args.folders, args.output_csv)
