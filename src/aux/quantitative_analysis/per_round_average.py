import pandas as pd
import ast
import json
import os
import argparse
from pathlib import Path

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
    base_rank_path = base_dir / 'rank_based' / strategy

    # Read data from files
    mds_df = pd.read_csv(
        os.path.join(base_mds_path, f'averaged_mds_results_{strategy}.csv'),
        sep=','
    )

    normal_df = pd.read_csv(
        os.path.join(base_rank_path, 'results--ver-43_1.csv'),
        sep=','
    )

    def parse_expositions(rec_str):
        """
        Convert a semicolon-separated string into a list of integers.
        Example: "1;23;35;2" -> [1, 23, 35, 2]
        """
        if pd.isna(rec_str) or rec_str.strip() == '':
            return []
        return list(map(int, rec_str.split(';')))

    # List of networks
    networks = ['aucs', 'ckm_physicians', 'eu_transportation', 'eu_transport_klm', 'lazega']

    # List of method pairs
    mds_methods = ['d^deg_c', 'd^deg_cd', 'd^nghb_sd', 'd^p_rnk', 'd^p_rnk_m', 'd^v_rnk', 'd^v_rnk_m']
    normal_methods = ['deg_c', 'deg_cd', 'nghb_sd', 'p_rnk', 'p_rnk_m', 'v_rnk', 'v_rnk_m']

    # List of mi_values and seed_budgets
    mi_values = [round(0.1 * i, 1) for i in range(1, 10)]

    # Define seed budgets per strategy
    seed_budgets_or = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    seed_budgets_and = [15, 20, 25, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    # Select based on strategy
    seed_budgets = seed_budgets_and if strategy == "and" else seed_budgets_or

    # Ensure correct data types
    mds_df['mi_value'] = mds_df['mi_value'].astype(float)
    mds_df['seed_budget'] = mds_df['seed_budget'].astype(int)
    normal_df['mi_value'] = normal_df['mi_value'].astype(float)
    normal_df['seed_budget'] = normal_df['seed_budget'].astype(int)

    mds_df['expositions_rec'] = mds_df['expositions_rec'].apply(ast.literal_eval)

    results = {network: {mds_method: {} for mds_method in mds_methods} for network in networks}

    # Iterate through each network
    for network in networks:
        print(f"Processing network: {network}")
        # Iterate through each pair of MDS and normal methods
        for mds_method, normal_method in zip(mds_methods, normal_methods):
            print(f"  Comparing method pair: MDS='{mds_method}' vs Rank-Based='{normal_method}'")
            # Iterate through each mi_value
            for mi_value in mi_values:
                # Iterate through each seed_budget
                for seed_budget in seed_budgets:
                    # Filter rows for the current combination in MDS dataset
                    mds_row = mds_df[
                        (mds_df['network'] == network) &
                        (mds_df['ss_method'] == mds_method) &
                        (mds_df['mi_value'] == mi_value) &
                        (mds_df['seed_budget'] == seed_budget)
                        ]
                    # Filter rows for the current combination in rank-based dataset
                    normal_row = normal_df[
                        (normal_df['network'] == network) &
                        (normal_df['ss_method'] == normal_method) &
                        (normal_df['mi_value'] == mi_value) &
                        (normal_df['seed_budget'] == seed_budget)
                        ]

                    # Proceed if both rows are found
                    if not mds_row.empty and not normal_row.empty:
                        # Assuming there's only one row per combination
                        mds_expositions = mds_row.iloc[0]['expositions_rec']
                        normal_expositions = parse_expositions(normal_row.iloc[0]['expositions_rec'])

                        # Determine the maximum number of spread rounds
                        max_rounds = max(len(mds_expositions), len(normal_expositions))

                        # Initialize lists to hold per-round differences
                        differences = []
                        for i in range(max_rounds):
                            mds_val = mds_expositions[i] if i < len(mds_expositions) else 0
                            normal_val = normal_expositions[i] if i < len(normal_expositions) else 0
                            diff = mds_val - normal_val
                            differences.append(diff)

                        # Store the differences in the results dictionary
                        # Keyed by mi_value and seed_budget
                        if mi_value not in results[network][mds_method]:
                            results[network][mds_method][mi_value] = {}
                        if seed_budget not in results[network][mds_method][mi_value]:
                            results[network][mds_method][mi_value][seed_budget] = differences
                    else:
                        # Handle missing rows if necessary
                        # For example, log missing combinations or assign NaN
                        pass  # Currently ignoring missing combinations

    output_file = f"per_round_average_{strategy}.json"

    # Save raw per round results to the file
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f"Results saved to {output_file}")

    # Average over seed budget
    average_over_seed_budget = {}

    for network_name, methods_dict in results.items():
        average_over_seed_budget[network_name] = {}
        for method, mi_values_dict in methods_dict.items():
            average_over_seed_budget[network_name][method] = {}
            for mi_value, seed_budgets_dict in mi_values_dict.items():
                # Collect all seed budget lists
                seed_budget_lists = list(seed_budgets_dict.values())

                if not seed_budget_lists:
                    # Handle empty seed budget lists
                    average_over_seed_budget[network_name][method][mi_value] = []
                    continue

                # Determine the maximum length among all seed budget lists
                max_length = max(len(lst) for lst in seed_budget_lists)

                # Pad shorter lists with zeros to match max_length
                padded_lists = [
                    lst + [0] * (max_length - len(lst)) for lst in seed_budget_lists
                ]

                # Initialize a list to store the sum of each element
                summed_elements = [0] * max_length

                # Sum each element across all padded lists
                for lst in padded_lists:
                    for idx, value in enumerate(lst):
                        summed_elements[idx] += value

                # Calculate the average for each element
                num_lists = len(padded_lists)
                averaged_list = [s / num_lists for s in summed_elements]

                # Store the averaged list
                average_over_seed_budget[network_name][method][mi_value] = averaged_list

    output_file = f"average_over_seed_budget_{strategy}.json"

    with open(output_file, 'w') as file:
        json.dump(average_over_seed_budget, file, indent=4)

    print(f"Results saved to {output_file}")

    # Further average over mi_value
    # This step averages across different mi_values for each network and method
    average_over_mi_value = {}

    for network_name, methods_dict in average_over_seed_budget.items():
        average_over_mi_value[network_name] = {}
        for method, mi_values_dict in methods_dict.items():
            # Collect all averaged lists across mi_values
            averaged_lists = list(mi_values_dict.values())

            if not averaged_lists:
                # Handle empty mi_values
                average_over_mi_value[network_name][method] = []
                continue

            # Determine the maximum length among all averaged lists
            max_length = max(len(lst) for lst in averaged_lists)

            # Pad shorter lists with zeros to match max_length
            padded_averaged_lists = [
                lst + [0] * (max_length - len(lst)) for lst in averaged_lists
            ]

            # Initialize a list to store the sum of each element
            summed_elements = [0] * max_length

            # Sum each element across all padded averaged lists
            for lst in padded_averaged_lists:
                for idx, value in enumerate(lst):
                    summed_elements[idx] += value

            # Calculate the average for each element
            num_lists = len(padded_averaged_lists)
            final_averaged_list = [s / num_lists for s in summed_elements]

            # Store the final averaged list
            average_over_mi_value[network_name][method] = final_averaged_list

    output_file = f"average_over_seed_budget_and_mi_value_{strategy}.json"
    with open(output_file, 'w') as file:
        json.dump(average_over_mi_value, file, indent=4)

    print(f"Results saved to {output_file}")

    # Further average over methods
    average_over_methods = {}

    for network_name, methods_dict in average_over_mi_value.items():
        # Collect all method-averaged lists for each network
        method_lists = list(methods_dict.values())

        if not method_lists:
            # Handle case where no methods exist for this network
            average_over_methods[network_name] = []
            continue

        # Determine the maximum length among all lists
        max_length = max(len(lst) for lst in method_lists)

        # Pad all lists with zeros to match the maximum length
        padded_lists = [
            lst + [0] * (max_length - len(lst)) for lst in method_lists
        ]

        # Perform element-wise averaging
        n = len(padded_lists)
        summed_elements = [0] * max_length
        for lst in padded_lists:
            for idx, value in enumerate(lst):
                summed_elements[idx] += value

        averaged_list = [s / n for s in summed_elements]

        # Store the final averaged list for this network
        average_over_methods[network_name] = averaged_list

    output_file = f"average_over_seed_budget_mi_value_and_method_{strategy}.json"
    with open(output_file, 'w') as file:
        json.dump(average_over_methods, file, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()