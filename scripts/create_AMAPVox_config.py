#!/usr/bin/env python
# -- coding: utf-8 --

"""
Script for automatically creating AMAPVox config files
"""

from pathlib import Path
import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

TLS = True
ULS = False

# Inputs
la_file = "H:/movingtree_b2h/data/metrics/mean_single_leaf_area.txt"

mean_la = pd.read_csv(la_file, sep=" ", usecols=[0, 3], index_col=0)

helios_playback_dir = "H:/helios/output/"
folder_pattern_uls = "uls_tree"
folder_pattern_tls = "tls_tree"

if ULS:
    f_template = "H:/movingtree_b2h/data/AMAPVox_config/uls_tree1_static.xml"
    tree = ET.parse(f_template)
    root = tree.getroot()

    folders = [f.path for f in os.scandir(helios_playback_dir) if f.is_dir() and folder_pattern_uls in f.path]
    for folder in folders:
        pc_dir = sorted(list(Path(folder).glob('*')), key=lambda file: file.stat().st_ctime, reverse=True)[0]
        survey_name = Path(pc_dir).parent.stem
        tree_id = int(survey_name.split("_")[1][4:])
        path_merged_pc = (Path(pc_dir) / f"{survey_name}.laz")
        outfile = f"H:/movingtree_b2h/data/AMAPVox_config/{survey_name}.xml"

        for scan in root.iter("scan"):
            scan.set("src", path_merged_pc.as_posix())
        for traj in root.iter("trajectory"):
            traj.set("src", path_merged_pc.as_posix().replace(".laz", "_trajectory.txt"))
        for out in root.iter("output"):
            out.set("src", path_merged_pc.as_posix().replace(".laz", ".vox"))
        for la in root.iter("single-leaf-area"):
            la.set("value", f"{mean_la.loc[tree_id].item()}")

        tree.write(outfile)

if TLS:
    f_template = "H:/movingtree_b2h/data/AMAPVox_config/tls_tree1_static.xml"
    tree = ET.parse(f_template)
    root = tree.getroot()

    folders = [f.path for f in os.scandir(helios_playback_dir) if f.is_dir() and folder_pattern_tls in f.path]
    for folder in folders:
        pc_dir = sorted(list(Path(folder).glob('*')), key=lambda file: file.stat().st_ctime, reverse=True)[0]
        scans = list(Path(pc_dir).glob("leg00?_points.laz"))
        trajectories = list(Path(pc_dir).glob("leg00?_trajectory.txt"))
        survey_name = Path(folder).stem
        tree_id = int(survey_name.split("_")[1][4:])
        for scan_file, traj_file in zip(scans, trajectories):
            with open(traj_file) as f:
                t_coords = f.read().strip().split(" ")[:3]
                t_coords = [float(c) for c in t_coords]
                # add 1.7 m for tripod (mounted at 1.5 m) and scanner VZ-400 (beam origin at 0.2 m)
                t_coords[2] += 1.7
            outfile = f"H:/movingtree_b2h/data/AMAPVox_config/{survey_name}_{scan_file.stem}.xml"
            print(outfile)
            voxfile = scan_file.as_posix().replace(".laz", ".vox")
            for scan in root.iter("scan"):
                scan.set("src", scan_file.as_posix())
            for sp in root.iter("scanner"):
                sp.set("position", f"({', '.join([str(coord) for coord in t_coords])})")
            for out in root.iter("output"):
                out.set("src", voxfile)
            for la in root.iter("single-leaf-area"):
                la.set("value", f"{mean_la.loc[tree_id].item()}")

            tree.write(outfile)
