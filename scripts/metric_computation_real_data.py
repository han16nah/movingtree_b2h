#!/usr/bin/env python
# -- coding: utf-8 --

# !/usr/bin/env python
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import laspy
from jakteristics import las_utils, compute_features
import open3d as o3d

from scripts import plot
from scripts import voxelizer

import metric_computation_tls

REGULAR = True
MLS = True
HEIGHT_METRICS = True
GEOMETRIC_FEATS = True
VOXEL_METRICS = True
LA_METRICS = True

mpl.use('Qt5Agg')

# Inputs
infiles = [Path("../data/real_trees/2022-05-11_TLS_tree4_t_filtered_classified.laz"),
           # Path("data/real_trees/2022-05-11_TLS_tree6_t_filtered_classified.laz")
           ]
path_cc_bin = "C:/Program Files/CloudCompare_2_13_alpha/CloudCompare.exe"

# Output locations
plot_dir = "../data/plots"
metrics_dir = "../data/metrics"
geom_feats_dir = "../data/metrics/geomfeats_real/"

Path(plot_dir).mkdir(parents=True, exist_ok=True)
Path(geom_feats_dir).mkdir(parents=True, exist_ok=True)

search_radii = [0.05, 0.01]  # for geometric features; maybe check again

# initiate geometric feature DataFrame
features_jakteristics = ['planarity', 'linearity', 'sphericity', 'verticality']

# initiate height metric DataFrame
height_metric_df = pd.DataFrame(columns=["no_points", "no_wood_points", "no_leaf_points",
                                         "Mean", "Max", "Std",
                                         "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"])

# initiate voxel DataFrame
vox_metrics = ['percentage_filled_vox', 'mean_ppv', 'median_ppv', 'max_ppv', 'std_ppv']
voxel_sizes = [0.1, 0.05, 0.02, 0.01]
vox_df = pd.DataFrame(columns=vox_metrics,
                      index=[int(v * 100) for v in voxel_sizes])

for path in infiles:
    paths_leaf = []
    paths_wood = []
    paths_merged = []
    if REGULAR:
        paths_leaf.append(path.as_posix().replace(".laz", "_leaves_sub.las"))
        paths_wood.append(path.as_posix().replace(".laz", "_wood_sub.las"))
        paths_merged.append("")
    if MLS:
        paths_leaf.append(Path(path.parent) / f"{path.stem}_leaves_sub_mls_raw_0.laz")
        paths_wood.append(Path(path.parent) / f"{path.stem}_wood_sub.las")
        paths_merged.append(Path(path.parent) / f"{path.stem}_mls_0.laz")

    print(paths_leaf)
    print(paths_wood)
    print(paths_merged)
    for path_leaves, path_wood, path_merged in zip(paths_leaf, paths_wood, paths_merged):
        if "mls" in str(path_leaves):
            suffix = "_mls"
        else:
            suffix = ""

        # read with laspy
        las_l = laspy.read(path_leaves)
        las_w = laspy.read(path_wood)
        coords_leaves = np.array([las_l.x, las_l.y, las_l.z]).T
        coords_wood = np.array([las_w.x, las_w.y, las_w.z]).T
        coords = np.vstack((coords_leaves, coords_wood))

        if not Path(path_merged).exists() and path_merged != "":
            command = ["lasmerge",
                       "-i", f"{Path(path_leaves).as_posix()}",
                       f"{Path(path_wood).as_posix()}",
                       "-o", f"{Path(path_merged).as_posix()}"]
            print(f"Executing command:\n{' '.join(command)}")
            p = subprocess.run(command)
        else:
            print(f"{Path(path_merged)} already exists.")

        survey_name = Path(path).stem
        tree_id = "_".join(survey_name.split("_")[0:3])

        if HEIGHT_METRICS:
            # Basic stats (on no. points, etc.)
            n_points = coords.shape[0]
            n_points_wood = coords_wood.shape[0]
            n_points_leaf = coords_leaves.shape[0]

            # Height and distribution metrics
            height_vals = coords[:, 2]
            qq = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
            hq = np.zeros(len(qq), dtype=float)
            for idx, qq in enumerate(qq):
                hq[idx] = np.percentile(height_vals, qq)

            height_metric_df.loc[tree_id, ["no_points", "no_wood_points", "no_leaf_points",
                                           "Mean", "Max", "Std"]] = [
                n_points,
                n_points_wood,
                n_points_leaf,
                np.nanmean(height_vals),
                np.nanmax(height_vals),
                np.nanstd(height_vals)]
            height_metric_df.loc[tree_id, ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]] = hq
            height_metric_df.to_pickle(Path(metrics_dir) / f"{path.stem}_height_metrics{suffix}.pkl")

        if GEOMETRIC_FEATS:
            # Geometric features
            for search_radius in search_radii:
                n_threads = 4
                dim_features = compute_features(coords_leaves, search_radius=search_radius, num_threads=n_threads,
                                                feature_names=features_jakteristics)
                dim_feats_df = pd.DataFrame(dim_features, columns=features_jakteristics)
                hist_fname = f"{path.stem}_leaves_{int(search_radius * 100)}_leaves_gf{suffix}.pkl"
                dim_feats_df.to_pickle(Path(geom_feats_dir) / hist_fname)

                dim_features_wood = compute_features(coords_wood, search_radius=search_radius, num_threads=n_threads,
                                                     feature_names=features_jakteristics)
                dim_feats_df_wood = pd.DataFrame(dim_features_wood, columns=features_jakteristics)
                hist_fname_wood = f"{path.stem}_{int(search_radius * 100)}_wood_gf{suffix}.pkl"
                dim_feats_df_wood.to_pickle(Path(geom_feats_dir) / hist_fname_wood)

                plot.plot_features_histogram(dim_features, features=features_jakteristics)
                # plt.show()
                plt.savefig(Path(plot_dir) / f"{path.stem}_{int(search_radius * 100)}_leaves_feature_hist{suffix}.png")
                plt.clf()

        if VOXEL_METRICS:
            # Voxel based features
            for vox_size in voxel_sizes:
                perc_filled, pts_per_vox = metric_computation_tls.voxel_metrics(coords, vox_size)

                j = int(vox_size * 100)
                vox_df.loc[j,
                ['percentage_filled_vox', 'mean_ppv', 'median_ppv', 'max_ppv', 'std_ppv']] = [
                    perc_filled,
                    np.mean(pts_per_vox),
                    np.median(pts_per_vox),
                    np.max(pts_per_vox),
                    np.std(pts_per_vox)
                ]
                vox_df.to_pickle(Path(metrics_dir) / f"{path.stem}_vox{suffix}.pkl")

        if LA_METRICS:
            infile = str(path_leaves).replace("_raw", "")
            max_edge_length = 0.04
            tri_area = metric_computation_tls.area_2_5_delaunay_cc(path_cc_bin, infile, max_edge_length)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords_leaves)

            # alpha shape leaf area
            alpha = 0.015
            alpha_shape_area = metric_computation_tls.area_alpha_shape(pcd, alpha)

            # Poisson surface reconstruction leaf area
            poisson_area = metric_computation_tls.area_poisson(pcd)

            la_df = pd.DataFrame(data={"alpha_leaf_area": [alpha_shape_area / 2],
                                       "poisson_leaf_area": [poisson_area],
                                       "tri_leaf_area": [tri_area]})

            la_df.to_pickle(Path(metrics_dir) / f"{path.stem}_la{suffix}.pkl")
