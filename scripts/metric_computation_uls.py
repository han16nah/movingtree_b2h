#!/usr/bin/env python
# -- coding: utf-8 --

"""
Computation of various metrics

Hannah Weiser, Heidelberg University, March 2023
"""

# Imports
import sys
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import laspy

from scripts import voxelizer
from scripts import metric_computation_tls

from typing import TypeVar, Union, Tuple, List
import numpy.typing as npt

PathLike = TypeVar(Union[str, bytes, os.PathLike])


def get_traj_uls(out_pc_dir: PathLike, z_trafo: float = 0.26) -> Union[npt.NDArray, List]:
    """
    Function to get a numpy array of the merged trajectory (and the names of the traj files) from
    a HELIOS++ survey output directory
    """
    traj_list = []
    leg_names = []
    for traj_file in Path(out_pc_dir).glob("leg*_trajectory.txt"):
        t_coords = np.genfromtxt(traj_file, delimiter=" ")
        # add 0.26 m for scanner mount (0.2 m) and scanner (beam origin at 0.06 m)
        t_coords[:, 2] += z_trafo
        traj_list.append(t_coords)
        leg_names.append(traj_file.stem.replace("_trajectory", ""))
    merged_traj = np.vstack(traj_list)

    return merged_traj, leg_names


if __name__ == "__main__":
    REGULAR = True
    MLS = True
    GEOMETRIC_FEATS = True
    HEIGHT_METRICS = True
    VOXEL_METRICS = True

    # Inputs
    helios_playback_dir = "H:/helios/output/"
    folder_pattern_uls = "uls_tree"

    # Output locations
    plot_dir = "../data/plots"
    metrics_dir = "../data/metrics"

    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    scenarios = ["static", "a", "b", "c"]
    tree_ids = [k for k in range(1, 15)]
    print(tree_ids)
    index = pd.MultiIndex.from_product([scenarios, tree_ids],
                                       names=["scenario", "tree_id"])

    search_radii = [0.1, 0.25]  # for geometric features; maybe check again

    # initiate voxel DataFrame
    vox_metrics = ['percentage_filled_vox', 'mean_ppv', 'median_ppv', 'max_ppv', 'std_ppv']
    voxel_sizes = [0.25, 0.1, 0.05]
    vm_index = pd.MultiIndex.from_product([scenarios, tree_ids, [int(v * 100) for v in voxel_sizes]],
                                          names=["scenario", "tree_id", "voxel_size"])
    vox_df = pd.DataFrame(columns=vox_metrics,
                          index=vm_index)

    # initiate height metric DataFrame
    height_metric_df = pd.DataFrame(columns=["no_points", "no_wood_points", "no_leaf_points",
                                             "Mean", "Max", "Std",
                                             "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
                                    index=index)

    folders = [f.path for f in os.scandir(helios_playback_dir) if f.is_dir() and folder_pattern_uls in f.path]

    if REGULAR:
        for folder in folders:

            # get last run directory
            pc_dir = metric_computation_tls.get_last_run_dir(folder)
            survey_name = Path(pc_dir).parent.stem
            tree_id = int(survey_name.split("_")[1][4:])
            path_merged_pc = (Path(pc_dir) / f"{survey_name}.laz")
            scenario = metric_computation_tls.get_scenario(survey_name)

            # Merge Point Cloud
            metric_computation_tls.merge_laz_if_not_exists(pc_dir, path_merged_pc)

            # get (merged) trajectory
            full_traj, all_traj = get_traj_uls(pc_dir, z_trafo=0.26)
            # and write trajectory to file
            if (Path(pc_dir) / f"{survey_name}_trajectory.txt").exists():
                print(f'{(Path(pc_dir) / f"{survey_name}_trajectory.txt").as_posix()} already exists. Overwriting ...')
            np.savetxt((Path(pc_dir) / f"{survey_name}_trajectory.txt"), full_traj, delimiter=" ", fmt="%.8f",
                       header="X Y Z GPSTime Roll Pitch Yaw", comments="")

            # read with laspy
            las = laspy.read(str(path_merged_pc))

            # get coordinates, classification and coordinates by classification
            coords = np.array([las.x, las.y, las.z]).T
            classification = np.array([las.classification])
            coords_leaves = coords[(classification == 1)[0], :]
            coords_wood = coords[(classification == 0)[0], :]
            coords_blossom = coords[(classification == 2)[0], :]

            if VOXEL_METRICS:
                # Voxel based features
                for vox_size in voxel_sizes:
                    perc_filled, pts_per_vox = metric_computation_tls.voxel_metrics(coords, vox_size,
                                                                                    target_count=1)

                    j = int(vox_size * 100)
                    vox_df.loc[
                        (scenario, tree_id, j), ['percentage_filled_vox', 'mean_ppv', 'median_ppv', 'max_ppv',
                                                 'std_ppv']] = [
                        perc_filled,
                        np.mean(pts_per_vox),
                        np.median(pts_per_vox),
                        np.max(pts_per_vox),
                        np.std(pts_per_vox)
                    ]

            if HEIGHT_METRICS:
                # Basic stats (on no. points, etc.)
                n_points = coords.shape[0]
                n_points_wood = coords_wood.shape[0]
                n_points_leaf = coords_leaves.shape[0] + coords_blossom.shape[0]

                # Height and distribution metrics
                height_vals = coords[:, 2]
                qq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
                hq = np.zeros(len(qq), dtype=float)
                for idx, qq in enumerate(qq):
                    hq[idx] = np.percentile(height_vals, qq)

                height_metric_df.loc[(scenario, tree_id), ["no_points", "no_wood_points", "no_leaf_points",
                                                           "Mean", "Max", "Std"]] = [
                    n_points,
                    n_points_wood,
                    n_points_leaf,
                    np.nanmean(height_vals),
                    np.nanmax(height_vals),
                    np.nanstd(height_vals)]
                height_metric_df.loc[(scenario, tree_id), ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]] = hq

        if VOXEL_METRICS:
            vox_df.to_pickle(Path(metrics_dir) / f"uls_voxel_metrics.pkl")
        if HEIGHT_METRICS:
            height_metric_df.to_pickle(Path(metrics_dir) / f"uls_height_metrics.pkl")

    for df in [vox_df, height_metric_df]:
        for col in df.columns:
            df[col].values[:] = 0
