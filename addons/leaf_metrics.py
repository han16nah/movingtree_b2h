import numpy as np
from pathlib import Path
import os
import math
from mathutils import Quaternion, Matrix, Euler, Vector
import bpy
import bmesh


def get_leaf_angles(leaf_objects):

    zeniths = np.empty(shape=(len(leaf_objects),))
    azimuths = np.empty(shape=(len(leaf_objects),))

    # get normal vectors of all leaves
    for i, leaf in enumerate(leaf_objects):
        assert(leaf.type == 'MESH')
                    
        # get leaf base
        rot_origin = leaf.matrix_world @ leaf.data.vertices[0].co
            
            
        # get normal vector
        # any face from the mesh will do, as leaves are all planar, so let's just get the first
        normal_vec = leaf.data.polygons[0].normal.to_4d()
        normal_vec.w = 0
        normal_vec = (leaf.matrix_world @ normal_vec).to_3d()
    
        # get zenith angle
        ang_zenith = math.degrees(normal_vec.angle(Vector((0,0,1))))
        ang_azimuth = math.degrees(normal_vec.angle(Vector((0,1,0))))
    
        if ang_zenith > 90:
            ang_zenith = 180-ang_zenith

        zeniths[i] = ang_zenith
        azimuths[i] = ang_azimuth
    
    return zeniths, azimuths


def get_leaf_area(leaf_objects):
    total_area = 0
    
    for obj in leaf_objects:
        print(obj.name)
        assert(obj.type == 'MESH')

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        area = sum(f.calc_area() for f in bm.faces)
        print(area)
        
        total_area += area
        
        bm.free()
    
    return total_area


if __name__ == "__main__":
    
    # get all leaves
    scene = bpy.context.scene
    objects = scene.objects
    
    leaves = []
    for o in objects:
        if o.name.startswith("leaves") or o.name.startswith("Leaves"):
        # if o.name.startswith("Plane"):
            leaves.append(o)

    zenith_angles, azimuth_angles = get_leaf_angles(leaves)
    
    output_directory = "H:/movingtree_b2h/data"
    
    project_name = Path(bpy.data.filepath).stem
    
    
    # save to file to later analyze in notebook
    np.savetxt(Path(output_directory) / f"{project_name}_leaf_zenith_angles.txt", zenith_angles, fmt='%.8f')
    np.savetxt(Path(output_directory) / f"{project_name}_leaf_azimuth_angles.txt", azimuth_angles, fmt='%.8f')

    leaf_area = get_leaf_area(leaves)
    
    print(f"{leaf_area} m2")