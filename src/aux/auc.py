"""A script with functions calculating AuC from the spreading dynamics."""

import numpy as np
import pandas as pd

from src.aux import slicer_plotter


def prepare_cdfs(mds_slice: pd.DataFrame, nml_slice: pd.DataFrame) -> dict[str, np.ndarray | int]:
    # compute raw cdf
    mds_cdf = slicer_plotter.ResultsSlicer.mean_expositions_rec(mds_slice)["cdf"]
    nml_cdf = slicer_plotter.ResultsSlicer.mean_expositions_rec(nml_slice)["cdf"]
    # pad the shorter one
    padding_size = max(len(nml_cdf), len(mds_cdf))
    nml_cdf = np.pad(
        nml_cdf,
        (0, np.abs(padding_size - len(nml_cdf))),
        "constant",
        constant_values=nml_cdf[-1],
    )
    mds_cdf = np.pad(
        mds_cdf,
        (0, np.abs(padding_size - len(mds_cdf))),
        "constant",
        constant_values=mds_cdf[-1],
    )
    # obtain max available value and discounting number
    nb_actors = (mds_slice["exposed_nb"] + mds_slice["unexposed_nb"]).iloc[0].item()
    seed_nb = nml_slice["seed_nb"].iloc[0].item()
    # return CDFs and min/max values
    return {
        "mds_cdf": mds_cdf,
        "nml_cdf": nml_cdf,
        "max_val": nb_actors,
        "start_val": seed_nb,
    }


def area_under_curve(cdf: np.ndarray, start_val: int, max_value: int) -> float:
    """Compute AuC from cdf in range [0,1] and discarded seed set impact."""
    if len(cdf) < 2:
        raise ValueError("CDF must co(ntain at least two values.")
    cdf_scaled = (cdf - start_val) / (max_value - start_val)
    cdf_steps = np.linspace(0, 1, len(cdf_scaled))
    return np.trapezoid(cdf_scaled, cdf_steps)
