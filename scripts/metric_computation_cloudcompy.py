#!/usr/bin/env python
# -- coding: utf-8 --

"""
Script to calculate leaf area from a point cloud.
Filters using SOR, then goes through points with different IDs, subsamples them, computes the 2.5D Delaunay
triangulation, gets the area, sums them up and saves the outputs for all input point clouds in a txt file
"""

from pathlib import Path
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import cloudComPy as cc

import numpy.typing as npt


def triangle_area_3d(v1: npt.NDArray, v2: npt.NDArray, v3: npt.NDArray) -> float:
    """Calculate the area of a triangle in 3D space.

    Parameters:
    v1, v2, v3 (numpy.ndarray): 3D vectors representing the vertices of the triangle.

    Returns:
    float: The area of the triangle.
    """
    # Calculate the two vectors corresponding to two edges of the triangle
    u = v2 - v1
    v = v3 - v1

    # Calculate the cross product of the two vectors
    cross_product = np.cross(u, v)

    # Calculate the magnitude of the cross product and divide by 2
    area = 0.5 * np.linalg.norm(cross_product)

    return area


# Inputs
helios_playback_dir = "H:/helios/output/"
folder_pattern_tls = "tls_tree"
folder_pattern_uls = "uls_tree"

# Output locations
metrics_dir = "../data/metrics"

folders = [f.path for f in os.scandir(helios_playback_dir) if f.is_dir() and folder_pattern_tls in f.path]

outfile1 = Path(metrics_dir) / "leaf_areas_tri.txt"
outfile2 = Path(metrics_dir) / "leaf_areas_tri_mls.txt"

with open(outfile1, "w") as f:
    f.write("scenario tree_id la_tri\n")
with open(outfile2, "w") as f:
    f.write("scenario tree_id la_tri_mls\n")

for folder in folders:
    pc_dir = sorted(list(Path(folder).glob('*')), key=lambda file: file.stat().st_ctime, reverse=True)[0]
    survey_name = Path(pc_dir).parent.stem
    tree_id = int(survey_name.split("_")[1][4:])
    path_merged_pc = (Path(pc_dir) / f"{survey_name}.laz")
    if survey_name.endswith("static"):
        scenario = "static"
    elif survey_name.endswith("_a"):
        scenario = "a"
    elif survey_name.endswith("_b"):
        scenario = "b"
    elif survey_name.endswith("_c"):
        scenario = "c"
    else:
        scenario = "invalid"
    # Merge and Load Point Cloud
    if not path_merged_pc.exists():
        command = ["lasmerge", "-i", f"{pc_dir}/*.laz", "-o", f"{path_merged_pc.as_posix()}", "-faf"]
        print(f"Executing command:\n{' '.join(command)}")
        p = subprocess.run(command)
    else:
        print(f"{path_merged_pc.as_posix()} already exists.")

    path_mls = (Path(pc_dir) / f"{survey_name}_leaves_mls_seg.laz")
    # load point cloud
    cloud = cc.loadPointCloud(path_merged_pc.as_posix())
    # filter by classification to obtain only leaf points
    dic = cloud.getScalarFieldDic()
    cloud.setCurrentOutScalarField(dic['Classification'])
    cl1 = cc.filterBySFValue(0.5, 1.0, cloud)
    # perform statistical outlier removal (SOR)
    cl1_filt = cc.CloudSamplingTools.sorFilter(cl1, knn=6, nSigma=1.0)
    (cl1_sor, res) = cl1.partialClone(cl1_filt)
    try:
        cl1_sor.setName("sor_cloud")
    except AttributeError:
        continue

    # load mls-filtered point cloud
    cl2 = cc.loadPointCloud(path_mls.as_posix())

    for cl, outf in zip([cl1_sor, cl2], [outfile1, outfile2]):
        with open(outf, "a") as f:
            # compute on leaf basis
            dic = cl.getScalarFieldDic()
            try:
                cl.setCurrentOutScalarField(dic['hitObjectId'])
                sf_object_id = cl.getScalarField(dic['hitObjectId'])
                asf_object_id = sf_object_id.toNpArray()
            except KeyError:
                cl.setCurrentOutScalarField(dic['Point Source ID'])
                sf_object_id = cl.getScalarField(dic['Point Source ID'])
                asf_object_id = sf_object_id.toNpArray()
            leaf_tri_areas = np.zeros(int(np.max(np.unique(asf_object_id))+1))

            for leaf_id in np.unique(asf_object_id):
                leaf_id = int(leaf_id)
                # print(leaf_id)
                # filter by leaf id
                cl_leaf = cc.filterBySFValue(leaf_id-0.1, leaf_id+0.1, cl)
                # subsample point cloud
                params = cc.SFModulationParams()
                cl_ss = cc.CloudSamplingTools.resampleCloudSpatially(cl_leaf, 0.004, params)
                (cl_sub, res) = cl_leaf.partialClone(cl_ss)
                cl_sub.setName("cl_sor_sub 4mm")

                # Triangulate the point cloud (Delaunay Tri, best fitting plane via least squares)
                mesh = cc.ccMesh.triangulate(cl_sub, cc.TRIANGULATION_TYPES.DELAUNAY_2D_BEST_LS_PLANE, updateNormals=True,
                                             maxEdgeLength=0.04)
                try:
                    mesh.setName("delaunay_mesh")
                except AttributeError:
                    print(f"Error for leaf ID {leaf_id}")
                    continue

                # Get triangle indices and coordinates as numpy arrays
                tri = mesh.IndexesToNpArray_copy()
                coords = cl_sub.toNpArrayCopy()

                # Compute total triangle area (as proxy for the one-sided leaf area of the tree)
                tri_areas = np.zeros(tri.shape[0])
                # loop over triangles
                for i, vertex in enumerate(tri):
                    v1 = coords[tri[i, 0]]
                    v2 = coords[tri[i, 1]]
                    v3 = coords[tri[i, 2]]

                    tri_areas[i] = triangle_area_3d(v1, v2, v3)

                leaf_tri_areas[leaf_id] = np.sum(tri_areas)

            total_tri_area = np.sum(leaf_tri_areas)
            print(total_tri_area)
            f.write(f"{scenario} {tree_id} {total_tri_area:.8f}\n")

            del total_tri_area
            del leaf_tri_areas
            del tri_areas
