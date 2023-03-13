#!/usr/bin/env python
# -- coding: utf-8 --

"""
Additional processing steps for HELIOS++ scenes

Hannah Weiser, 02-2023
"""

# Imports
from pathlib import Path
import shutil


def add_mtl_file(path_obj, path_mtl, mtl_idx):
    obj_dir = Path(path_obj).parent

    try:
        rel_path_mtl = Path(path_mtl).absolute().relative_to(obj_dir)
    except:
        # copy over
        new_path_mtl = Path(obj_dir) / Path(path_mtl).name
        shutil.copy(path_mtl, new_path_mtl)
        rel_path_mtl = new_path_mtl.relative_to(obj_dir)

    # get material names from mtl file
    material_names = []
    with open(path_mtl, "r") as mtl_file:
        for line in mtl_file:
            if line.startswith("newmtl"):
                material_names.append(line.strip().split(" ")[1])

    lines = []
    with open(path_obj, "r") as f:
        content = f.read()
        if "mtllib" in content and "usemtl" in content:
            print(f"Aborting. File {path_obj} already associated with a material library.")
        else:
            f.seek(0)
            for line in f:
                if line.startswith("mtllib"):
                    # leave out already defined mtllib (if no usemtl present)
                    pass
                if line.startswith("f"):
                    lines.append(f"mtllib {rel_path_mtl}\n")
                    lines.append(f"usemtl {material_names[mtl_idx]}\n")
                    lines.append(line)
                    break
                lines.append(line)
            remaining_lines = f.readlines()
            lines += remaining_lines
            with open(path_obj, "w") as outf:
                outf.writelines(lines)
                print(f"Written material {material_names[mtl_idx]} to file {path_obj}")


if __name__ == "__main__"   :
    helios_root = "H:/helios"
    obj_paths = list((Path(helios_root) / "data/sceneparts").glob("tree*/*.obj"))
    mtl_path = "../data/leafwood.mtl"

    for obj_file in obj_paths:
        if "leaves" in obj_file.stem or "Leaves" in obj_file.stem:
            add_mtl_file(obj_file, mtl_path, 0)
        elif "tree" in obj_file.stem:
            add_mtl_file(obj_file, mtl_path, 1)
        else:
            print(f"Skipping {obj_file.stem}.")
