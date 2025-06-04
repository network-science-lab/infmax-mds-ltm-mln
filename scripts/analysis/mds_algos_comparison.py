"""Statistical analysis of MDS rankings obtained with LI and Greedy algos."""

from pathlib import Path

import pandas as pd
from src.aux import slicer_plotter
from src.loaders.net_loader import load_network


def generate_similarities_mds(workdir: Path) -> None:
    print("plotting statistics on differences between MDS obtained with greedy and local iprov.")
    batches = [
        "batch_1",
        "batch_2",
        "batch_3",
        "batch_4",
        "batch_5",
        "batch_6",
        "batch_7",
        "batch_8",
        "batch_9",
        "batch_10",
        "batch_11",
        "batch_12",
    ]
    used_mds_list = []
    for batch_id in batches:
        print(batch_id)
        used_mds_list.extend(
            slicer_plotter.JSONParser().read_minimal_dominating_sets(
                f"data/raw_results/{batch_id}/rankings.zip"
            )
        )
    used_mds_df = pd.DataFrame(used_mds_list)

    # get types of MDS used for comparison and MDS lengths
    used_mds_df["mds_algo"] = used_mds_df["ss_method"].apply(lambda x: x.split("^")[0])
    used_mds_df["ss_method"] = used_mds_df["ss_method"].apply(lambda x: x.split("^")[-1])
    used_mds_df["mds_len"] = used_mds_df["mds"].apply(lambda x: len(x))

    # obtain avg and std of differences of MDSs obtained greadily and with LI across the same nets
    # and repetitions
    _df = used_mds_df.pivot_table(
        index=['network', 'version'],
        columns='mds_algo',
        values='mds_len'
    ).dropna(subset=['d', 'D'])
    _df["diff"] = _df["d"] - _df["D"]
    final_df = _df.groupby("network")["diff"].agg(avg_diff="mean", std_diff="std").reset_index()

    # normalise numbers by network sizes
    actors_nbs = {}
    for net_name in final_df["network"]:
        net_graph = load_network(net_name, as_tensor=False)
        actors_nbs[net_name] = net_graph.get_actors_num()
    final_df = final_df.set_index("network")
    final_df.loc[:, "net_size"] = actors_nbs
    final_df["std_diff"] /= final_df["net_size"]
    final_df["avg_diff"] /= final_df["net_size"]
    final_df = final_df.drop("net_size", axis=1)
    final_df.to_csv(workdir / "g_vs_li_mds.csv")

    # prepare latex representation
    final_df["avg_difference"] = (
        "$" +
        final_df["avg_diff"].apply(lambda x: f"{x:.2f}") +  
        " (" +
        final_df["std_diff"].apply(lambda x: f"{x:.0E}") +
        ")$"
    )
    final_df["avg_difference"].to_latex("table.txt")



if __name__ == "__main__":

    # prepare outout directory
    root_dir = Path(__file__).resolve().parent.parent.parent
    workdir = root_dir / "data/processed_results"
    workdir.mkdir(exist_ok=True, parents=True)

    # compute the DFs
    mds_similarity_df = generate_similarities_mds(workdir)
