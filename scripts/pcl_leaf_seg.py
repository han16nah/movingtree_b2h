import os
import sys
from pathlib import Path
import pclpy
from pclpy import pcl
import open3d as o3d
import numpy as np
import subprocess
import itertools
import shutil
from typing import List


def dict_permute(my_dict: dict) -> List[dict]:
    """
    Function which takes a dictionary with lists of values (parameters) for the keys and permutes them.
    The output is a list of dictionaries, one for each parameter combination.
    """
    keys, values = zip(*my_dict.items())
    permutation_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutation_dicts


if __name__ == '__main__':

    tls = True
    uls = False

    infiles_tls = Path("H:/helios/output/").glob("tls_tree*/*/tls_tree*.laz")
    exclude = ["mls", "wood", "blossom"]
    infiles = [f for f in infiles_tls if not any(substr in f.as_posix() for substr in exclude)]
    # dictionary can multiple entries per value for testing different settings (all combinations will be run)
    settings = {"mls_search_radius": [0.05],
                "normal_search_radius": [0.1],  # [0.05, 0.1],
                "rg_min_size": [10],
                "rg_max_size": [150000],
                "rg_n_neighbours": [50],  # [30, 50],
                "rg_smooth_thresh": [2],  # [1, 2],  # in degrees
                "rg_curvature_thresh": [1],
                "rg_residual_thresh": [1]}
    settings_dicts = dict_permute(settings)

    for f in infiles:
        print(f.as_posix())
        f_leaf = f.as_posix().replace(".laz", "_leaves.las")

        pc = pclpy.read(f_leaf, "PointXYZ")

        path_settings = f.parent / "settings.txt"
        with open(path_settings, "w") as outf_settings:
            for j, s in enumerate(settings_dicts):
                print("using settings:")
                print(s)

                pc_out = pclpy.moving_least_squares(pc,
                                                    search_radius=s["mls_search_radius"],
                                                    compute_normals=True,
                                                    polynomial_order=1,
                                                    num_threads=4)

                normals = pcl.PointCloud.Normal()
                outf_mls = f_leaf.replace(".las", f"_mls.laz")
                # if not Path(outf_mls).exists():
                pclpy.write_las(pc_out, outf_mls)
                pc_out.compute_normals(radius=s["normal_search_radius"], output_cloud=normals, num_threads=8)

                clusters = pclpy.region_growing(pc_out,
                                                normals=normals,
                                                min_size=s["rg_min_size"],
                                                max_size=s["rg_max_size"],
                                                n_neighbours=s["rg_n_neighbours"],
                                                smooth_threshold=s["rg_smooth_thresh"],
                                                curvature_threshold=s["rg_curvature_thresh"],
                                                residual_threshold=s["rg_residual_thresh"])

                print(f"Number of cluster: {len(clusters)}")

                cluster_pc_list = []
                cols = np.zeros((pc_out.xyz.shape[0], 3))

                outfolder = Path(f_leaf).parent / f"mls_leaf_clusters_{j}"
                if outfolder.exists():
                    shutil.rmtree(outfolder)
                outfolder.mkdir()

                for i, cluster in enumerate(clusters):
                    idx = np.array(cluster.indices)

                    points = pc_out.xyz[idx, :]
                    pcd_cluster = pcl.PointCloud.PointXYZ.from_array(points)

                    outf_name = Path(f_leaf).name.replace(".las", f"_mls_cluster_{i}.laz")
                    outf = outfolder / outf_name
                    pclpy.write_las(pcd_cluster, outf)
                    print(f"Written {outf}")

                # save unsegmented point cloud
                outf = f_leaf.replace(".las", "_mls.laz")
                pclpy.write_las(pc_out, outf)
                print(f"Written {outf}")

                # save segmented point cloud (by merging segments stored in cluster subfolder)
                path_mls_segmented = f_leaf.replace(".las", f"_mls_{j}.laz")
                # path_mls_segmented = f_leaf.replace(".las", f"_mls_seg.laz")
                command = ["lasmerge", "-i", f"{outfolder.as_posix()}/*_mls_cluster_{j}.laz", "-o",
                           f"{Path(path_mls_segmented).as_posix()}", "-faf"]
                print(f"Executing command:\n{' '.join(command)}")
                p = subprocess.run(command)
                outf_settings.write(f"{j} {s}\n")
