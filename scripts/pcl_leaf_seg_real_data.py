import os
import sys
from pathlib import Path
import pclpy
from pclpy import pcl
import open3d as o3d
import numpy as np
import subprocess
import laspy
import filter
import shutil
import itertools


def get_las_data_type(array):
    las_data_types = [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
        # "S",  # strings not implemented
    ]
    type_ = str(array.dtype)
    if type_ not in las_data_types:
        raise NotImplementedError("Array type not implemented: %s" % type_)
    return las_data_types.index(type_) + 1


def get_offset(filepath):
    with laspy.file.File(filepath) as f:
        return f.header.offset


def get_dims(las_file):
    dimensions = []
    for dimension in las_file.point_format:
        dimensions.append(dimension.name)
    return sorted(dimensions)


def read_las(in_path, xyz_offset=None):
    if xyz_offset is None:
        xyz_offset = np.array([0, 0, 0])
    with laspy.file.File(in_path) as f:
        supported_attrs = "intensity Reflectance Deviation pt_src_id"
        all_attrs = get_dims(f)
        wanted_attrs = [a for a in all_attrs if a in supported_attrs]
        pc_data = np.zeros((f.header.count, 3 + len(wanted_attrs)))
        pc_data[:, :3] = np.array([f.x, f.y, f.z]).T  # - xyz_offset
        dimensions = ["x", "y", "z"]
        for ind, attr in enumerate(wanted_attrs):
            val = f.points['point'][attr]
            pc_data[:, ind + 3] = val
            dimensions.append(attr)

    return pc_data, dimensions


def translate_pc(pc_data, translation):
    pc_data[:, :3] = pc_data[:, :3] + translation


def ndarray_to_pcl(array, point_type):
    assert point_type in "PointXYZ PointXYZI PointXYZINormal PointNormal PointXYZRGBNormal PointXYZRGBA"
    testcloud = pcl.PointCloud.PointXYZ()
    pcl_pc = testcloud.from_array(array)
    return pcl_pc


def dict_permute(my_dict: dict):
    keys, values = zip(*my_dict.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutation_dicts


if __name__ == '__main__':

    setting_versions = {"mls_search_radius": [0.1],
                        "mls_poly_order": [1],
                        "normal_search_radius": [0.1],
                        "rg_min_size": [10],
                        "rg_max_size": [1000000],
                        "rg_n_neighbours": [50],  # 30? 60?
                        "rg_smooth_thresh": [2],  # in degrees
                        "rg_curvature_thresh": [2],
                        "rg_residual_thresh": [1]}

    BY_SP = False
    ALL_SP = True

    infiles = [
        Path("H:/movingtree_b2h/data/real_trees/2022-05-11_TLS_tree4_t_filtered_classified.laz"),
        # Path("H:/movingtree_b2h/data/real_trees/2022-05-11_TLS_tree6_t_filtered_classified.laz")
    ]
    setting_log = Path("H:/movingtree_b2h/data/real_trees/settings.txt")
    settings_permutation = dict_permute(setting_versions)

    for path in infiles:
        path_leaf = path.as_posix().replace(".laz", "_leaves_sub.las")
        path_wood = path.as_posix().replace(".laz", "_wood_sub.las")
        data, dims = read_las(path_leaf)
        n_scan_positions = int(np.max(data[:, dims.index("pt_src_id")]))

        if ALL_SP:
            pc = pclpy.read(path_leaf, "PointXYZ")
            with open(setting_log, "w") as setting_f:
                for i, settings in enumerate(settings_permutation):
                    print("Running with settings:")
                    print(settings)
                    pc_out = pclpy.moving_least_squares(pc,
                                                        search_radius=settings["mls_search_radius"],
                                                        compute_normals=True,
                                                        polynomial_order=settings["mls_poly_order"],
                                                        num_threads=4)
                    normals = pcl.PointCloud.Normal()
                    pc_out.compute_normals(radius=settings["normal_search_radius"], output_cloud=normals,
                                           num_threads=8)  # 0.1

                    pclpy.write_las(pc_out, path_leaf.replace(".las", f"_mls_raw_{i}.laz"))

                    clusters = pclpy.region_growing(pc_out,
                                                    normals=normals,
                                                    min_size=settings["rg_min_size"],
                                                    max_size=settings["rg_max_size"],
                                                    n_neighbours=settings["rg_n_neighbours"],
                                                    smooth_threshold=settings["rg_smooth_thresh"],
                                                    curvature_threshold=settings["rg_curvature_thresh"],
                                                    residual_threshold=settings["rg_residual_thresh"])

                    print(f"Number of cluster: {len(clusters)}")

                    outfolder = Path(path_leaf).parent / f"{'_'.join(path.stem.split('_')[:3])}_mls_leaf_clusters_{i}"
                    if outfolder.exists():
                        shutil.rmtree(outfolder)
                    outfolder.mkdir()

                    for j, cluster in enumerate(clusters):
                        idx = np.array(cluster.indices)
                        points = pc_out.xyz[idx, :]
                        pcd_cluster = pcl.PointCloud.PointXYZ.from_array(points)
                        outf_name = Path(path_leaf).name.replace(".las", f"_mls_cluster_{j}.laz")
                        outf = outfolder / outf_name
                        pclpy.write_las(pcd_cluster, outf)
                        print(f"Written {outf}")

                    # regular (no mls) version needs different settings!!
                    normals2 = pcl.PointCloud.Normal()
                    pc.compute_normals(radius=settings["normal_search_radius"], output_cloud=normals2,
                                       num_threads=8)  # 0.1
                    clusters2 = pclpy.region_growing(pc,
                                                     normals=normals2,
                                                     min_size=settings["rg_min_size"],
                                                     max_size=settings["rg_max_size"],
                                                     n_neighbours=settings["rg_n_neighbours"],
                                                     smooth_threshold=5,
                                                     curvature_threshold=2,
                                                     residual_threshold=1)
                    print(f"Number of cluster: {len(clusters2)}")

                    cols = np.zeros((pc.xyz.shape[0], 3))
                    outfolder2 = Path(path_leaf).parent / f"{'_'.join(path.stem.split('_')[:3])}_leaf_clusters_{i}"
                    if outfolder2.exists():
                        shutil.rmtree(outfolder2)
                    outfolder2.mkdir()
                    for j, cluster in enumerate(clusters2):
                        idx = np.array(cluster.indices)
                        points2 = pc.xyz[idx, :]
                        pcd_cluster2 = pcl.PointCloud.PointXYZ.from_array(points2)
                        outf2_name = Path(path_leaf).name.replace(".las", f"_cluster_{j}.laz")
                        outf2 = outfolder2 / outf2_name
                        pclpy.write_las(pcd_cluster2, outf2)
                        print(f"Written {outf2}")

                    # save mls version
                    path_mls_segmented = path_leaf.replace(".las", f"_mls_{i}.laz")
                    command = ["lasmerge", "-i", f"{outfolder.as_posix()}/*_mls_cluster*.laz", "-o",
                               f"{Path(path_mls_segmented).as_posix()}", "-faf"]
                    print(f"Executing command:\n{' '.join(command)}")
                    subprocess.run(command)

                    path_mls = path.as_posix().replace(".laz", f"_mls_{i}.laz")
                    command = ["lasmerge", "-i",
                               f"{path_mls_segmented}",
                               f"{path_wood}",
                               "-o",
                               f"{Path(path_mls).as_posix()}", "-faf"]
                    print(f"Executing command:\n{' '.join(command)}")
                    subprocess.run(command)
                    setting_f.write(f"{i} {settings}\n")

                    # save regular
                    path_segmented = path_leaf.replace(".las", f"_{i}.laz")
                    command = ["lasmerge", "-i", f"{outfolder2.as_posix()}/*_cluster*.laz", "-o",
                               f"{Path(path_segmented).as_posix()}", "-faf"]
                    print(f"Executing command:\n{' '.join(command)}")
                    subprocess.run(command)

                    path_merged = path.as_posix().replace(".laz", f"_{i}.laz")
                    command = ["lasmerge", "-i",
                               f"{path_segmented}",
                               f"{path_wood}",
                               "-o",
                               f"{Path(path_merged).as_posix()}", "-faf"]
                    print(f"Executing command:\n{' '.join(command)}")
                    subprocess.run(command)
