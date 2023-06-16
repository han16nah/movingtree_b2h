# !/usr/bin/env python
# -- coding: utf-8 --

"""
Computation of various metrics

Hannah Weiser, Heidelberg University, June 2023
"""
# Imports
import sys
import os
from pathlib import Path
import subprocess
import shutil

import numpy as np
import pandas as pd
import laspy
from jakteristics import las_utils, compute_features
import open3d as o3d

from scripts import voxelizer
from typing import TypeVar, Union, Tuple, List
import numpy.typing as npt

PathLike = TypeVar(Union[str, bytes, os.PathLike])


def get_scenario(name_survey: str) -> str:
    """Return the motion scneario based on the survey name"""
    if name_survey.endswith("static"):
        motion_scenario = "static"
    elif name_survey.endswith("_a"):
        motion_scenario = "a"
    elif name_survey.endswith("_b"):
        motion_scenario = "b"
    elif name_survey.endswith("_c"):
        motion_scenario = "c"
    else:
        motion_scenario = "invalid"

    return motion_scenario


def merge_laz_if_not_exists(pc_directory: PathLike, merged_pc_path: PathLike) -> None:
    """Merge all laz files in a directory LAStools (if the output path does not exist yet)"""
    merged_pc_path = Path(merged_pc_path)
    if not merged_pc_path.exists():
        cmd = ["lasmerge", "-i", f"{pc_directory}/*.laz", "-o", f"{merged_pc_path.as_posix()}", "-faf"]
        print(f"Executing command:\n{' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print(f"{merged_pc_path.as_posix()} already exists.")


def get_last_run_dir(directory: PathLike) -> str:
    """Get the latest HELIOS++ run directory in the output folder of a specific survey"""
    return sorted(list(Path(directory).glob('*')), key=lambda file: file.stat().st_ctime, reverse=True)[0]


def get_traj(out_pc_dir: PathLike, z_trafo: float) -> Union[List[npt.NDArray], List]:
    """Read and merge the trajectory files in a HELIOS++ output folder"""
    trajectory_list = []
    all_leg_names = []
    for traj_file in Path(out_pc_dir).glob("*_trajectory.txt"):
        with open(traj_file) as f:
            t_coords = f.read().strip().split(" ")[:3]
            t_coords = [float(c) for c in t_coords]
            # add trafo according to scanner mount and beam origin
            t_coords[2] += z_trafo
        trajectory_list.append(np.array(t_coords))
        all_leg_names.append(traj_file.stem.replace("_trajectory", ""))

    return trajectory_list, all_leg_names


def voxel_metrics(coordinates: npt.NDArray, voxel_size: float, target_count: int = 3) -> Tuple[float, npt.NDArray]:
    """
    Function to compute voxel metrics (percentage of filled voxels and an array with the number of points per voxels)
    from coordinates (np.array)
    """
    vox = voxelizer.Voxelizer(coordinates, voxel_size=voxel_size)
    centers, idxs, voxel_idx, closest_idx, local_origin = vox.voxelize()
    mins = np.min(voxel_idx, axis=0)
    maxes = np.max(voxel_idx, axis=0)
    x_range = int((maxes[0] - mins[0]) / 1) + 1
    y_range = int((maxes[1] - mins[1]) / 1) + 1
    z_range = int((maxes[2] - mins[2]) / 1) + 1
    # prepare some coordinates
    voxels = np.zeros((x_range, y_range, z_range), bool)
    voxel_idx_f, idxs_f = voxelizer.filter_by_point_count(voxel_idx, idxs, target_count=target_count)
    for idx in voxel_idx_f:
        idxlist = (idx - mins).astype(int).tolist()
        voxels[idxlist[0], idxlist[1], idxlist[2]] = True

    percentage_filled = np.count_nonzero(voxels) / np.size(voxels) * 100
    points_per_vox = np.array([len(idxs_f[i]) for i in range(len(voxel_idx_f))])

    return percentage_filled, points_per_vox


def area_alpha_shape(pcd_leaves, alpha_o3d: float) -> float:
    """Function to compute the area of an alpha shape of a point cloud"""
    alpha_shape_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_leaves, alpha_o3d)
    alpha_shape_area = alpha_shape_mesh.get_surface_area()

    return alpha_shape_area


def area_poisson(pcd_leaves, normal_search_radius: float = 0.05) -> float:
    """Function to perform Poisson surface reconstruction and compute the area of the resulting surface mesh"""
    # estimate normals
    pcd_leaves.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamRadius(radius=normal_search_radius)
    )
    pcd_leaves.orient_normals_consistent_tangent_plane(k=6)

    # run poisson surface reconstruction
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_leaves, depth=10)
    densities = np.asarray(densities)

    # use mean of densities as threshold to remove vertices (cf. Boukhana et al. 2022)
    vertices_to_remove = densities < np.mean(densities)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    area = mesh.get_surface_area()

    return area


def spatial_subsampling_cc(cc_bin: PathLike, incloud_path: PathLike, subcloud_path: PathLike, space: float):
    cmd = [cc_bin,
           "-SILENT",
           "-O", incloud_path,
           "-AUTO_SAVE", "OFF",
           "-SS", "SPATIAL", str(float(space)),
           "-C_EXPORT_FMT", "LAS",
           "-SAVE_CLOUDS", "FILE", subcloud_path]
    subprocess.run(cmd)


def area_2_5_delaunay_cc(cc_bin: PathLike, incloud_path: PathLike, max_edge_length: float, delete_files: bool = True):
    subsampled_path = Path(incloud_path).as_posix().replace(".las", "_sub.las").replace(".laz", "_sub.las")

    spatial_subsampling_cc(cc_bin, incloud_path, subsampled_path, space=0.004)

    las = laspy.read(subsampled_path)
    try:
        sf_name = "hitObjectId"
        n_ids = np.max(las.points[sf_name])
    except ValueError:
        sf_name = "Point Source ID"
        n_ids = np.max(las.points["point_source_id"])

    outpath_list = []
    area = 0
    outfolder = Path(incloud_path).parent / "tmp"
    outfolder.mkdir(exist_ok=True)
    for sp_id in range(int(n_ids) + 1):
        # print(sp_id)
        outmesh_path = outfolder / f"{sp_id}.ply"
        outpath_list.append(outmesh_path)
        cmd = [cc_bin,
               "-SILENT",
               "-O", str(subsampled_path),
               "-AUTO_SAVE", "OFF",
               "-SET_ACTIVE_SF", f"{sf_name}",  # needs CC 2.13 --> download and test again
               "-FILTER_SF", str(sp_id - 0.1), str(sp_id + 0.1),
               "-DELAUNAY",
               "-BEST_FIT",
               "-MAX_EDGE_LENGTH", str(max_edge_length),
               "-M_EXPORT_FMT", "PLY",
               "-SAVE_MESHES", "FILE", str(outmesh_path)]
        subprocess.run(cmd)
        mesh = o3d.io.read_triangle_mesh(str(outmesh_path))
        area += mesh.get_surface_area()

    if delete_files:
        os.remove(subsampled_path)
        shutil.rmtree(outfolder)

    return area


if __name__ == "__main__":
    REGULAR = True
    MLS = True
    GEOMETRIC_FEATS = True
    HEIGHT_METRICS = True
    VOXEL_METRICS = True
    LA_METRICS = True

    # Inputs
    helios_playback_dir = "H:/helios/output/"
    folder_pattern_tls = "tls_tree"
    path_cc_bin = "C:/Program Files/CloudCompare_2_13_alpha/CloudCompare.exe"

    # Output locations
    metrics_dir = "../data/metrics"
    geom_feats_dir = "../data/metrics/geomfeats/"

    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(geom_feats_dir).mkdir(parents=True, exist_ok=True)

    scenarios = ["static", "a", "b", "c"]
    tree_ids = [k for k in range(1, 16)]
    index = pd.MultiIndex.from_product([scenarios, tree_ids],
                                       names=["scenario", "tree_id"])

    search_radii = [0.05, 0.01]  # for geometric features; maybe check again
    # geometric features to compute
    features_jakteristics = ['planarity', 'linearity', 'sphericity', 'verticality']

    # initiate voxel DataFrame
    vox_metrics = ['percentage_filled_vox', 'mean_ppv', 'median_ppv', 'max_ppv', 'std_ppv']
    voxel_sizes = [0.1, 0.05, 0.02, 0.01]
    vm_index = pd.MultiIndex.from_product([scenarios, tree_ids, [int(v * 100) for v in voxel_sizes]],
                                          names=["scenario", "tree_id", "voxel_size"])
    vox_df = pd.DataFrame(columns=vox_metrics,
                          index=vm_index)

    # initiate height metric DataFrame
    height_metric_df = pd.DataFrame(columns=["no_points", "no_wood_points", "no_leaf_points",
                                             "Mean", "Max", "Std",
                                             "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"],
                                    index=index)

    # initiate LAI DataFrame
    la_df = pd.DataFrame(columns=["alpha_leaf_area", "poisson_leaf_area", "tri_leaf_area"],
                         index=index)

    folders = [f.path for f in os.scandir(helios_playback_dir) if f.is_dir() and folder_pattern_tls in f.path]
    print(folders)

    # load from previous runs, if existing
    try:
        if VOXEL_METRICS:
            vox_df = pd.read_pickle(Path(metrics_dir) / f"tls_voxel_metrics.pkl")
        if HEIGHT_METRICS:
            height_metric_df = pd.read_pickle(Path(metrics_dir) / f"tls_height_metrics.pkl")
        if LA_METRICS:
            la_df = pd.read_pickle(Path(metrics_dir) / f"tls_lai_gap_frac.pkl")
    except FileNotFoundError:
        print("INFO: One or more pickled dataframes do not yet exist.")

    if REGULAR:
        for folder in folders:
            pc_dir = get_last_run_dir(folder)

            survey_name = Path(pc_dir).parent.stem
            tree_id = int(survey_name.split("_")[1][4:])
            path_merged_pc = (Path(pc_dir) / f"{survey_name}.laz")

            scenario = get_scenario(survey_name)

            merge_laz_if_not_exists(pc_dir, path_merged_pc)

            traj_list, leg_names = get_traj(pc_dir, z_trafo=1.7)

            # read with laspy
            las = laspy.read(str(path_merged_pc))

            # get coordinates, classification and coordinates by classification
            coords = np.array([las.x, las.y, las.z]).T
            classification = np.array([las.classification])
            coords_leaves = coords[(classification == 1)[0], :]
            coords_wood = coords[(classification == 0)[0], :]
            coords_blossom = coords[(classification == 2)[0], :]

            if GEOMETRIC_FEATS:
                # Geometric features
                for search_radius in search_radii:
                    n_threads = 4
                    dim_features = compute_features(coords_leaves, search_radius=search_radius, num_threads=n_threads,
                                                    feature_names=features_jakteristics)
                    dim_feats_df = pd.DataFrame(dim_features, columns=features_jakteristics)
                    hist_fname = f"{scenario}_{tree_id}_{int(search_radius * 100)}_gf.pkl"
                    dim_feats_df.to_pickle(Path(geom_feats_dir) / hist_fname)

                    dim_features_wood = compute_features(coords_wood, search_radius=search_radius,
                                                         num_threads=n_threads,
                                                         feature_names=features_jakteristics)
                    dim_feats_df_wood = pd.DataFrame(dim_features_wood, columns=features_jakteristics)
                    hist_fname_wood = f"{scenario}_{tree_id}_{int(search_radius * 100)}_wood_gf.pkl"
                    dim_feats_df_wood.to_pickle(Path(geom_feats_dir) / hist_fname_wood)

            if VOXEL_METRICS:
                # Voxel based features
                for vox_size in voxel_sizes:
                    perc_filled, pts_per_vox = voxel_metrics(coords, vox_size)

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

            if LA_METRICS:
                # 2.5D Delaunay leaf area
                infile = str(path_merged_pc).replace(".laz", "_leaves.las")
                max_edge_length = 0.04
                tri_area = area_2_5_delaunay_cc(path_cc_bin, infile, max_edge_length)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords_leaves)

                # alpha shape leaf area
                alpha = 0.015
                alpha_shape_area = area_alpha_shape(pcd, alpha)

                # Poisson surface reconstruction leaf area
                poisson_area = area_poisson(pcd)
                print(f"For reference: Poisson area is {poisson_area}")
                la_df.loc[(scenario, tree_id), :] = [
                    alpha_shape_area / 2,
                    poisson_area,
                    tri_area
                ]

        if VOXEL_METRICS:
            vox_df.to_pickle(Path(metrics_dir) / f"tls_voxel_metrics.pkl")
        if HEIGHT_METRICS:
            height_metric_df.to_pickle(Path(metrics_dir) / f"tls_height_metrics.pkl")
        if LA_METRICS:
            la_df.to_pickle(Path(metrics_dir) / f"tls_la.pkl")

    ###################################################################################################################
    ###################################################################################################################
    # the same for the MLS-filtered point clouds

    # first delete all values in dataframes (from first run)
    for df in [vox_df, height_metric_df, la_df]:
        for col in df.columns:
            df[col].values[:] = 0

    if MLS:
        # load from previous runs, if existing
        try:
            if VOXEL_METRICS:
                vox_df = pd.read_pickle(Path(metrics_dir) / f"tls_voxel_metrics_mls.pkl")
            if HEIGHT_METRICS:
                height_metric_df = pd.read_pickle(Path(metrics_dir) / f"tls_height_metrics_mls.pkl")
            if LA_METRICS:
                lai_gap_frac_df = pd.read_pickle(Path(metrics_dir) / f"tls_lai_gap_frac_mls.pkl")
        except FileNotFoundError:
            print("INFO: One or more pickled dataframes do not yet exist.")

        for folder in folders:
            # get last run directory
            pc_dir = sorted(list(Path(folder).glob('*')), key=lambda file: file.stat().st_ctime, reverse=True)[0]
            survey_name = Path(pc_dir).parent.stem
            tree_id = int(survey_name.split("_")[1][4:])
            path_leaves = (Path(pc_dir) / f"{survey_name}_leaves_mls.laz")
            path_wood = (Path(pc_dir) / f"{survey_name}_wood.las")
            path_blossom = (Path(pc_dir) / f"{survey_name}_blossom.las")
            path_merged_pc = (Path(pc_dir) / f"{survey_name}_mls.laz")
            scenario = get_scenario(survey_name)

            # Merge Point Cloud
            if not path_merged_pc.exists():
                command0 = ["las2las", "-i", f"{path_leaves.as_posix()}",
                            "-set_version", "1.4", "-set_point_type", "6",
                            "-o", f"{path_leaves.as_posix().replace('.laz', '_14.laz')}"]
                print(f"Executing command:\n{' '.join(command0)}")
                p = subprocess.run(command0)
                command = ["lasmerge",
                           "-i", f"{path_leaves.as_posix().replace('.laz', '_14.laz')}",
                           f"{path_wood.as_posix()}",
                           f"{path_blossom.as_posix()}",
                           "-o", f"{path_merged_pc.as_posix()}", "-faf"]
                print(f"Executing command:\n{' '.join(command)}")
                p = subprocess.run(command)
            else:
                print(f"{path_merged_pc.as_posix()} already exists.")

            traj_list, leg_names = get_traj(pc_dir, z_trafo=1.7)

            # read with laspy
            las = laspy.read(str(path_merged_pc))

            # get coordinates and coordinates by classification
            coords = np.array([las.x, las.y, las.z]).T

            las_l = laspy.read(str(path_leaves))
            coords_leaves = np.array([las_l.x, las_l.y, las_l.z]).T

            las_w = laspy.read(str(path_wood))
            coords_wood = np.array([las_w.x, las_w.y, las_w.z]).T

            if path_blossom.exists():
                las_b = laspy.read(str(path_blossom))
                coords_blossom = np.array([las_b.x, las_b.y, las_b.z]).T
            else:
                coords_blossom = np.array([])

            if GEOMETRIC_FEATS:
                # Geometric features
                for search_radius in search_radii:
                    n_threads = 4
                    dim_features = compute_features(coords_leaves, search_radius=search_radius, num_threads=n_threads,
                                                    feature_names=features_jakteristics)
                    dim_feats_df = pd.DataFrame(dim_features, columns=features_jakteristics)
                    hist_fname = f"{scenario}_{tree_id}_{int(search_radius * 100)}_gf_mls.pkl"
                    dim_feats_df.to_pickle(Path(geom_feats_dir) / hist_fname)

                    dim_features_wood = compute_features(coords_wood, search_radius=search_radius,
                                                         num_threads=n_threads,
                                                         feature_names=features_jakteristics)
                    dim_feats_df_wood = pd.DataFrame(dim_features_wood, columns=features_jakteristics)
                    hist_fname_wood = f"{scenario}_{tree_id}_{int(search_radius * 100)}_wood_mls_gf.pkl"
                    dim_feats_df_wood.to_pickle(Path(geom_feats_dir) / hist_fname_wood)

            if VOXEL_METRICS:
                # Voxel based features
                for vox_size in voxel_sizes:
                    perc_filled, pts_per_vox = voxel_metrics(coords, vox_size)

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

            if LA_METRICS:
                # 2.5D Delaunay leaf area
                infile = str(path_leaves).replace(".laz", "_seg.laz")
                max_edge_length = 0.04
                tri_area = area_2_5_delaunay_cc(path_cc_bin, infile, max_edge_length)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(coords_leaves)

                # alpha shape leaf area
                alpha = 0.015
                alpha_shape_area = area_alpha_shape(pcd, alpha)

                # Poisson surface reconstruction leaf area
                poisson_area = area_poisson(pcd)

                la_df.loc[(scenario, tree_id), :] = [
                    alpha_shape_area / 2,
                    poisson_area,
                    tri_area
                ]

        if VOXEL_METRICS:
            vox_df.to_pickle(Path(metrics_dir) / f"tls_voxel_metrics_mls.pkl")
        if HEIGHT_METRICS:
            height_metric_df.to_pickle(Path(metrics_dir) / f"tls_height_metrics_mls.pkl")
        if LA_METRICS:
            la_df.to_pickle(Path(metrics_dir) / f"tls_la_mls.pkl")
