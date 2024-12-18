import pandas as pd
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process MDS and normal data for specified strategy.')
    parser.add_argument('--strategy', choices=['or', 'and'], required=True, help='Strategy to use: "or" or "and".')
    args = parser.parse_args()

    strategy = args.strategy

    # Define the base directories
    base_dir = r'C:\Users\Mingshan\PycharmProjects\infmax-mds-ltm-mln\raw_results_2\data\raw_results'

    # Paths for MDS-based results
    base_mds_path = os.path.join(base_dir, 'mds_based', strategy, 'small_real')

    # Paths for normal (rank-based) results
    normal_results_file = os.path.join(base_dir, 'rank_based', strategy, 'results--ver-43_1.csv')

    # The numerical columns
    numerical_columns = ['gain', 'simulation_length', 'seed_nb', 'exposed_nb', 'unexposed_nb']

    # Columns to keep as they are
    keep_columns = ['network', 'protocol', 'seed_budget', 'mi_value', 'ss_method']

    # Initialize an empty list to store DataFrames
    df_list = []

    # Build the file pattern and get matched files
    pattern = os.path.join(base_mds_path, 'results--ver-43_*.csv')
    matched_files = glob.glob(pattern)
    print(f"Matched files: {matched_files}")

    # Read and collect the DataFrames
    for filename in matched_files:
        df = pd.read_csv(filename, sep=',')  # Adjust 'sep' if necessary
        df_list.append(df)

    # Concatenate all DataFrames
    all_data = pd.concat(df_list, ignore_index=True)

    # Drop the 'seed_ids' column if it exists
    if 'seed_ids' in all_data.columns:
        all_data = all_data.drop(columns=['seed_ids'])

    # Group and aggregate the data
    result = all_data.groupby(keep_columns).agg({col: 'mean' for col in numerical_columns}).reset_index()

    # Save the result to a new CSV file
    output_path = os.path.join(base_mds_path, 'averaged_results.csv')
    result.to_csv(output_path, index=False)

    print(f"Processing complete. The file has been saved at: {output_path}")

    # Now, read data from files
    mds_results_file = output_path

    # Read data from files
    mds_df = pd.read_csv(mds_results_file, sep=',')
    normal_df = pd.read_csv(normal_results_file, sep=',')

    # List of networks
    networks = ['aucs', 'ckm_physicians', 'eu_transportation', 'eu_transport_klm', 'lazega']

    # List of method pairs
    mds_methods = ['d^deg_c', 'd^deg_cd', 'd^nghb_sd', 'd^p_rnk', 'd^p_rnk_m', 'd^random', 'd^v_rnk', 'd^v_rnk_m']
    normal_methods = ['deg_c', 'deg_cd', 'nghb_sd', 'p_rnk', 'p_rnk_m', 'random', 'v_rnk', 'v_rnk_m']

    # List of mi_values and seed_budgets
    mi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seed_budgets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    # Ensure correct data types
    mds_df['mi_value'] = mds_df['mi_value'].astype(float)
    mds_df['seed_budget'] = mds_df['seed_budget'].astype(int)
    normal_df['mi_value'] = normal_df['mi_value'].astype(float)
    normal_df['seed_budget'] = normal_df['seed_budget'].astype(int)

    print(mds_df)
    # Initialize results dictionary
    results = {}

    for network in networks:
        results[network] = {}
        for mds_method, normal_method in zip(mds_methods, normal_methods):
            wins = 0
            losses = 0
            equalities = 0

            for mi_value in mi_values:
                for seed_budget in seed_budgets:
                    # Filter rows for the current combination
                    mds_row = mds_df[
                        (mds_df['network'] == network) &
                        (mds_df['ss_method'] == mds_method) &
                        (mds_df['mi_value'] == mi_value) &
                        (mds_df['seed_budget'] == seed_budget)
                    ]
                    normal_row = normal_df[
                        (normal_df['network'] == network) &
                        (normal_df['ss_method'] == normal_method) &
                        (normal_df['mi_value'] == mi_value) &
                        (normal_df['seed_budget'] == seed_budget)
                    ]

                    # Proceed if both rows are found
                    if not mds_row.empty and not normal_row.empty:
                        mds_gain = mds_row.iloc[0]['gain']
                        normal_gain = normal_row.iloc[0]['gain']

                        if mds_gain > normal_gain:
                            wins += 1
                        elif mds_gain < normal_gain:
                            losses += 1
                        else:
                            equalities += 1
                    else:
                        # Skip if data is missing
                        continue

            # Store the result for the method pair
            results[network][(mds_method, normal_method)] = [wins, losses, equalities]

    # Output the results
    for network in networks:
        print(f"Network: {network}")
        for method_pair, counts in results[network].items():
            mds_method, normal_method = method_pair
            print(f"{mds_method} vs {normal_method}: {counts}")
        print()

    # Save the results to a file
    results_output_file = os.path.join(base_dir, f'results_{strategy}.txt')
    with open(results_output_file, 'w') as f:
        for network in networks:
            f.write(f"Network: {network}\n")
            for method_pair, counts in results[network].items():
                mds_method, normal_method = method_pair
                f.write(f"{mds_method} vs {normal_method}: {counts}\n")
            f.write("\n")
    print(f"Results have been saved to {results_output_file}")

if __name__ == '__main__':
    main()
