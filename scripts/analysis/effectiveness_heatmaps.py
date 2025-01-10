"""A script to procude effectiveness heatmaps."""

import itertools
from pathlib import Path
from typing import Optional, Union

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
    boundaries = np.linspace(vrange[0], vrange[1], n_levels)
    norm = BoundaryNorm(boundaries, ncolors=n_levels+1, extend="both")
    custom_cmap = ListedColormap(sns.color_palette(cmap, n_colors=n_levels))

    # continous color map
    # custom_cmap = sns.color_palette(cmap, as_cmap=True)

    # extreme values
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
    idle_df.loc[:, "comparison_gain"] = -2
    idle_df.loc[:, "comparison_auc"] = -2

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
    )[["mi_value", "seed_budget", metric]]
    heatmap_df = pd.pivot_table(
        full_combinations,
        index="mi_value",
        columns="seed_budget",
        values=metric,
        dropna=False,
    )
    return heatmap_df


def plot_heatmaps(quantitative_comparison_path: Path, out_dir: Path) -> None:

    print("loading data")
    df = read_quantitative_comparison(quantitative_comparison_path)

    print("plotting  heatmaps")
    plotter = slicer_plotter.ResultsPlotter()
    pdf = PdfPages(out_dir.joinpath(f"heatmaps.pdf"))

    for _, page_case in enumerate(plotter.yield_heatmap_config()):
        print(page_case)
        ssm, protocol, net_type = page_case

        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(5, 4),
            gridspec_kw={"width_ratios": [48, 48, 4]},
        )
        fig.tight_layout(pad=0.05, rect=(0.03, 0.10, 0.90, 0.87))

        gain_heatmap_df = prepare_heatmap(df, ssm, protocol, net_type, "comparison_gain")
        plot_heatmap(vis_df=gain_heatmap_df, heatmap_ax=axs[0], bar_ax=axs[2], vrange=(-1, 1))

        auc_heatmap_df = prepare_heatmap(df, ssm, protocol, net_type, "comparison_auc")
        plot_heatmap(vis_df=auc_heatmap_df, heatmap_ax=axs[1], bar_ax=axs[2], vrange=(-1, 1))

        fig.suptitle(
            f"MDS vs NML: difference (gain - left, auc - right)\n"
            f"{net_type}, {protocol}, {ssm}"
        )
        fig.savefig(pdf, format="pdf")
        plt.close(fig)

    pdf.close()


if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent.parent
    out_dir = root_path / "data/processed_results"
    out_dir.mkdir(exist_ok=True, parents=True)

    plot_heatmaps(root_path / "data/processed_results/quantitative_comparison.csv", out_dir)
