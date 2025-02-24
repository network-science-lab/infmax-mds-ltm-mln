"""A script to procude effectiveness heatmaps."""

import itertools
from pathlib import Path
from typing import Optional, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.aux import slicer_plotter


def prepare_ticklabels(series: pd.Index) -> Union[np.ndarray, str]:
    try:
        return series.to_numpy().round(2)
    except:
        return "auto"


def quantise_cmap(cmap: str, n_levels: int) -> ListedColormap:
    custom_cmap = ListedColormap(sns.color_palette(cmap, n_colors=n_levels))
    custom_cmap.set_bad(color="gray")
    # custom_cmap.set_over(color="gray")
    custom_cmap.set_under(color="gray")
    return custom_cmap


def plot_heatmap(
    vis_df: pd.DataFrame,
    heatmap_ax: plt.Axes,
    bar_ax: plt.Axes,
    vrange=(0., 1.),
    cmap="RdYlGn",
    mask: Optional[pd.DataFrame] = None,
    fmt: Optional[str] = ".2f",
) -> None:

    # quantised color map
    n_levels = 5
    custom_cmap = quantise_cmap(cmap, n_levels)
    boundaries = np.linspace(vrange[0], vrange[1], n_levels)
    norm = BoundaryNorm(boundaries, ncolors=n_levels+1, extend="both")

    # continous color map
    # custom_cmap = sns.color_palette(cmap, as_cmap=True)

    # extreme values
    vis_df = vis_df.replace(-1*float("inf"), -2)
    custom_cmap.set_under("black")
    custom_cmap.set_bad("gray")

    sns.heatmap(
        vis_df,
        ax=heatmap_ax,
        cbar_ax=bar_ax,
        cmap=custom_cmap,
        vmin=vrange[0],
        vmax=vrange[1],
        annot=True,
        annot_kws={"size": 7, "color": "black"},
        fmt=fmt,
        yticklabels=prepare_ticklabels(vis_df.index),
        xticklabels=prepare_ticklabels(vis_df.columns),
        linewidth=.5,
        mask=mask,
        cbar= True if bar_ax is not None else False,
    )
    heatmap_ax.invert_yaxis()
    heatmap_ax.collections[0].colorbar.set_ticks(boundaries)


def read_quantitative_comparison(file_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(file_path, index_col=0)

    # scale down gain to be in the same range as auc
    df_raw["mds_gain"] = df_raw["mds_gain"] / 100
    df_raw["nml_gain"] = df_raw["nml_gain"] / 100

    # drop records where there was no diffusion
    idle_df = df_raw.loc[(df_raw["mds_gain"] == 0.) & (df_raw["nml_gain"] == .0)].copy()
    efficient_df = df_raw.drop(idle_df.index)

    # assig values from the outside of the feasible range to mark cases where diffusion didn's start
    idle_df.loc[:, "comparison_gain"] = -1 * float("inf")
    idle_df.loc[:, "comparison_auc"] = -1 * float("inf")

    # add columns used for preparing heatmaps
    efficient_df.loc[:, "comparison_gain"] = efficient_df["mds_gain"] - efficient_df["nml_gain"]
    efficient_df.loc[:, "comparison_auc"] = efficient_df["mds_auc"] - efficient_df["nml_auc"]

    df = pd.concat([efficient_df, idle_df], ignore_index=True).reset_index(drop=True)
    return df[
        [
            "protocol",
            "mi_value",
            "seed_budget",
            "network", 
            "ss_method",
            "comparison_gain",
            "comparison_auc",
        ]
    ]


def map_network_types(df: pd.DataFrame) -> pd.DataFrame:
    df["network"] = df["network"].map(lambda x: slicer_plotter.ResultsPlotter._networks_groups[x])
    return df


def prepare_heatmap(
    df: pd.DataFrame,
    ssm: str,
    protocol: str,
    net_type: str,
    metric: str
) -> pd.DataFrame:
    """Prepare heatmap by creating an empyt df with all parame, copying real ones and pivoting."""
    df_case = df.loc[
        (df["ss_method"] == ssm) &
        (df["protocol"] == protocol) &
        (df["network"] == net_type)
    ]
    if protocol == "AND":
        budget_range = slicer_plotter.ResultsPlotter._seed_budgets_and
    else:
        budget_range = slicer_plotter.ResultsPlotter._seed_budgets_or
    full_combinations = pd.DataFrame(
        itertools.product(
            [protocol],
            slicer_plotter.ResultsPlotter._mi_values,
            budget_range,
            [net_type],
            [ssm],
            [np.nan],
            [np.nan],
        ), columns=df.columns
    )
    full_combinations = full_combinations.merge(
        df_case,
        on=["protocol", "mi_value", "seed_budget", "network", "ss_method"],
        how="left",
        suffixes=["_templ", ""],
    )[["mi_value", "seed_budget", metric]].rename(columns={"mi_value": "μ", "seed_budget": "s"})
    heatmap_df = pd.pivot_table(
        full_combinations,
        index="μ",
        columns="s",
        values=metric,
        dropna=False,
    )
    return heatmap_df


def plot_heatmaps_detailed(quantitative_comparison_path: Path, out_dir: Path) -> None:

    print("loading data")
    df = read_quantitative_comparison(quantitative_comparison_path)

    print("plotting  heatmaps")
    plotter = slicer_plotter.ResultsPlotter()
    pdf = PdfPages(out_dir.joinpath(f"heatmaps.pdf"))

    for _, page_case in enumerate(plotter.yield_heatmap_config()):
        print(page_case)
        ssm, protocol, (net_name, net_type) = page_case

        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(5, 4),
            gridspec_kw={"width_ratios": [48, 48, 4]},
        )
        fig.tight_layout(pad=0.05, rect=(0.03, 0.10, 0.90, 0.87))

        gain_heatmap_df = prepare_heatmap(df, ssm, protocol, net_name, "comparison_gain")
        plot_heatmap(vis_df=gain_heatmap_df, heatmap_ax=axs[0], bar_ax=axs[2], vrange=(-1, 1))

        auc_heatmap_df = prepare_heatmap(df, ssm, protocol, net_name, "comparison_auc")
        plot_heatmap(vis_df=auc_heatmap_df, heatmap_ax=axs[1], bar_ax=axs[2], vrange=(-1, 1))

        fig.suptitle(
            f"MDS vs NML: difference (gain - left, auc - right)\n"
            f"{net_name}, {protocol}, {ssm}"
        )
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()


def stack_heatmaps(chunk: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:

    ref_index = chunk[0].index.sort_values()
    ref_columns = chunk[0].columns.sort_values()

    chunk_list = []
    for chunk_df in chunk:
        chunk_list.append(chunk_df.reindex(index=ref_index, columns=ref_columns).to_numpy())
    chunk_arr = np.stack(chunk_list)
    
    too_small_mds_counts = np.sum(np.isnan(chunk_arr), axis=0)
    no_diffusion_counts = np.sum(chunk_arr == -np.inf, axis=0)
    all_counts = (np.ones_like(no_diffusion_counts) * chunk_arr.shape[0])
    feasible_counts = all_counts - (no_diffusion_counts + too_small_mds_counts)
    feasible_significant_counts = np.sum(
        ((chunk_arr > 0.01) & (chunk_arr <= 1)) | ((chunk_arr < -0.01) & (chunk_arr >= -1)),
        axis=0
    )
    feasible_nonsignificant_counts = feasible_counts - feasible_significant_counts
    aux_arr = (
        "(" + feasible_significant_counts.astype(str) + "/" + 
        feasible_nonsignificant_counts.astype(str) + ")/\n" +
        no_diffusion_counts.astype(str) + "/" +
        too_small_mds_counts.astype(str)
    )

    _chunk_arr = chunk_arr.copy()
    _chunk_arr[_chunk_arr == -np.inf] = np.nan
    avg_metric_diff = np.nanmean(_chunk_arr, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        feasible_mds_wins_prct = (100 * np.sum(chunk_arr > 0.01, axis=0)) / feasible_significant_counts
    feasible_mds_wins_prct[np.isnan(feasible_mds_wins_prct)] = -20

    return {
        "feasible_nodiff_toosmall_counts": pd.DataFrame(aux_arr, columns=ref_columns, index=ref_index),
        "avg_metric_diff": pd.DataFrame(avg_metric_diff, columns=ref_columns, index=ref_index),
        "feasible_mds_wins_prct": pd.DataFrame(feasible_mds_wins_prct, columns=ref_columns, index=ref_index),
    }


def plot_heatmaps_aggregated(quantitative_comparison_path: Path, out_dir: Path) -> None:

    print("loading data")
    df = read_quantitative_comparison(quantitative_comparison_path)
    
    print("creating empty dict for results slices")
    plotter = slicer_plotter.ResultsPlotter()
    df_dict = {}
    for net_type in  {*plotter._networks_groups.values()}:
        for protocol in {plotter._protocol_and, plotter._protocol_or}:
            for metric in {"gain", "auc"}:
                df_dict[(net_type, protocol, metric)] = []

    print("preparing heatmaps in the aggregated form")
    for _, page_case in enumerate(plotter.yield_heatmap_config()):
        print(page_case)
        ssm, protocol, (net_name, net_type) = page_case
        gain_heatmap_df = prepare_heatmap(df, ssm, protocol, net_name, "comparison_gain")
        df_dict[(net_type, protocol, "gain")].append(gain_heatmap_df)
        auc_heatmap_df = prepare_heatmap(df, ssm, protocol, net_name, "comparison_auc")
        df_dict[(net_type, protocol, "auc")].append(auc_heatmap_df)
    
    print("plotting  heatmaps")
    pdf = PdfPages(out_dir.joinpath(f"heatmaps_aggregated.pdf"))
    for exp_params, exp_results in df_dict.items():
        print(exp_params)
        exp_heatmap = stack_heatmaps(exp_results)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=(3.5, 5),
            gridspec_kw={"width_ratios": [95, 5]},
        )
        ax[0].set_aspect("equal")
        sns.heatmap(
            exp_heatmap["feasible_mds_wins_prct"],
            ax=ax[0],
            annot=exp_heatmap["feasible_nodiff_toosmall_counts"],
            annot_kws={"size": 8, "va": "top", "color": "black"},
            fmt="",
            cbar=False,
        )
        sns.heatmap(
            exp_heatmap["feasible_mds_wins_prct"],
            ax=ax[0],
            annot=exp_heatmap["avg_metric_diff"],
            annot_kws={"size": 10, "va": "bottom", "color": "black"},
            fmt=".2f",
            cbar=False,
        )
        sns.heatmap(
            exp_heatmap["feasible_mds_wins_prct"],
            ax=ax[0],
            cbar_ax=ax[1],
            annot=False,
            vmin=0,
            vmax=100,
            linecolor="black",
            linewidths=.5,
            cmap=quantise_cmap("RdYlGn", 11),
            cbar_kws={"shrink": .8},
            norm=BoundaryNorm(np.arange(0, 101, 10), ncolors=11, extend="min"),
        )
        fig.suptitle(
            f"networks: {exp_params[0].upper()}, " + 
            f"δ: {exp_params[1]}, " + 
            f"metric: {'Γ' if exp_params[2] == 'gain' else 'Λ'}"
        )
        fig.tight_layout(pad=0, rect=(-.05, 0, 1, .99))
        fig.savefig(pdf, format="pdf")
        plt.close(fig)
    pdf.close()

if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent.parent
    out_dir = root_path / "data/processed_results"
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_heatmaps_detailed(out_dir / "quantitative_comparison.csv", out_dir)
    plot_heatmaps_aggregated(out_dir / "quantitative_comparison.csv", out_dir)
