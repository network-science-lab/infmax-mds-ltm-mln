import pandas as pd
import numpy as np
import glob
import os
import sys
from pathlib import Path
import argparse


# Split and process the `expositions_rec` column


def main():

    parser = argparse.ArgumentParser(description='Process MDS and normal data for specified strategy.')
    parser.add_argument('--strategy', choices=['or', 'and'], required=True, help='Strategy to use: "or" or "and".')
    parser.add_argument('--base_dir', type=str, default=None, help='Base directory for raw results.')
    args = parser.parse_args()

    strategy = args.strategy

    # Determine the base directory
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        # Set default base_dir relative to the script's location
        script_dir = Path(__file__).parent.parent
        base_dir = script_dir / 'raw_results_2' / 'data' / 'raw_results'

    if not base_dir.exists():
        raise FileNotFoundError(f"The base directory {base_dir} does not exist.")

    # Paths for MDS-based results
    base_mds_path = base_dir / 'mds_based' / strategy / 'small_real'

    # Define the file pattern
    pattern = os.path.join(base_mds_path, 'results--ver-43_*.csv')

    # Print for verification
    print("Base MDS Path:", base_mds_path)
    print("Pattern:", pattern)

    # Columns to process and keep
    numerical_columns = ['gain', 'simulation_length', 'seed_nb', 'exposed_nb', 'unexposed_nb']
    keep_columns = ['network', 'protocol', 'seed_budget', 'mi_value', 'ss_method']

    # Initialize an empty list to store DataFrames
    df_list = []

    # Read all matched files
    for filename in glob.glob(pattern):
        df = pd.read_csv(filename, sep=',')  # Adjust 'sep' if necessary
        df_list.append(df)

    # Concatenate all DataFrames
    all_data = pd.concat(df_list, ignore_index=True)

    # Drop seed_ids column as it's not required
    all_data = all_data.drop(columns=['seed_ids'])

    def process_expositions(column):
        max_length = max(column.apply(lambda x: len(x.split(';')) if isinstance(x, str) else 0))
        processed_data = column.apply(lambda x: [int(i) for i in x.split(';')] if isinstance(x, str) else [])
        padded_data = processed_data.apply(lambda x: x + [0] * (max_length - len(x)))
        return np.array(padded_data.tolist())

    # Calculate element-wise averages for `expositions_rec` per group
    def average_expositions(group):
        indices = group.index
        expositions = expositions_array[indices]
        return expositions.mean(axis=0)

    # Process and pad `expositions_rec`
    expositions_array = process_expositions(all_data['expositions_rec'])

    # Group and aggregate other numerical columns
    grouped = all_data.groupby(keep_columns)
    aggregated = grouped.agg({col: 'mean' for col in numerical_columns}).reset_index()

    # Add the averaged expositions to the result
    averaged_expositions = grouped.apply(average_expositions, include_groups=False)  # Exclude grouping columns
    averaged_expositions_df = pd.DataFrame(
        averaged_expositions.tolist(),
        columns=[f'exposition_{i + 1}' for i in range(len(averaged_expositions.iloc[0]))]
    )
    result = pd.concat([aggregated, averaged_expositions_df], axis=1)

    # Combine all exposition columns into a single list column
    exposition_cols = [col for col in result.columns if col.startswith('exposition_')]
    result['expositions_rec'] = result[exposition_cols].values.tolist()

    # Drop the individual exposition_* columns
    result = result.drop(columns=exposition_cols)

    output_path = os.path.join(base_mds_path, f'averaged_mds_results_{strategy}.csv')
    result.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()

