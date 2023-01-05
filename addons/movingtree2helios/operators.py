
import bpy
import os
import sys
import numpy as np
import math
from mathutils import Quaternion, Matrix, Euler, Vector
from pathlib import Path
from bpy_extras.io_utils import ImportHelper, ExportHelper
from movingtree2helios import scene_writer as sw


def export_obj(self, context):
    # Deselect all objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False) 

    sceneparts_path = Path(self.helios_root) / (f"data/sceneparts/{self.sceneparts_folder}")
    sceneparts_path.mkdir(parents=True, exist_ok=True)
    
    filepaths_relative = []
    # Iterate over all objects and export them
    objects = bpy.context.view_layer.objects
    for ob in objects:
        objects.active = ob
        ob.select_set(True)

        if ob.type == 'MESH':
            outfile = str(sceneparts_path / (ob.name + '.obj'))
            
            if 'leaves.obj' in outfile: 
                outfile = outfile.replace('leaves.obj', 'leaves.000.obj')
            filepaths_relative.append(Path(outfile).relative_to(self.helios_root))
            bpy.ops.export_scene.obj(filepath=outfile, use_selection=True, axis_up='Z', axis_forward='Y', use_materials=False)

        ob.select_set(False)
    
    return filepaths_relative
    

def write_scene(self, context, obj_paths_relative):
    
    obj_paths_dynamic = []
    obj_paths_static = []
    sceneparts = ""
    
    
    # get translation that HELIOS will apply
    # get the global coordinates of all object bounding box corners    
    coords = np.vstack(
    tuple(np_matmul_coords(np.array(o.bound_box), o.matrix_world.copy())
         for o in  
            context.scene.objects
            if o.type == 'MESH'
            )
        )
    print("#" * 72)
    # bottom front left (all the mins)
    bfl = coords.min(axis=0)
    print(bfl)
    # convert to vector
    helios_shift = Vector(bfl)
    helios_shift = Vector((-0.380767, -0.359708, -0.000033))

    
    objects = bpy.context.view_layer.objects
    objects = [ob for ob in objects if ob.type == 'MESH']
    
    for i, ob in enumerate(objects):
        frames = []
        rotations = []
        try:
            for f in ob.animation_data.action.fcurves:
                for k in f.keyframe_points:
                    fr = k.co[0]
                    bpy.context.scene.frame_set(int(fr))
                    frames.append(int(fr))
                    rotations.append(ob.rotation_euler.copy().freeze())
                break
        except AttributeError:
            # ignore attribute error - happens on objects without animation data, e.g., the tree stem
            pass
        # check if moving object (i.e., rotations are different between keyframes)
        if len(set(rotations)) > 1:
            sceneparts += "\n\t\t<!--Dynamic scenepart-->"
            dynm_string = ""
            leaf_id = str(obj_paths_relative[i]).split("\\")[-1].replace(".obj", "")
            path = str(obj_paths_relative[i])
            prev_frame = 0
            rot_centre = ob.location
            j = 0
            prev_rot = Euler((0, 0, 0)).to_quaternion()
            for frame, rot in zip(frames, rotations):
                # determines number of loops
                frame_diff = frame - prev_frame
                if frame_diff == 0:
                    j += 1
                    continue
                
                # get rotation between actual and previous
                # current rotation to Quaternion
                q_rot = rot.to_quaternion()

                # subtract rotations -> multiply by inverse
                new_rot = q_rot @ prev_rot.inverted()
                
                # new rotation to axis angle
                axis, angle = new_rot.to_axis_angle()
                axis = axis[:]
                
                # determine if to continue with next dmotion or with first one (i.e., restart loop)
                if j+1 == len(frames):
                    next_id = 1
                    prev_frame = 0
                    prev_rot = Euler((0, 0, 0)).to_quaternion()
                else:
                    next_id = j+1
                    prev_frame = frame
                    prev_rot = rot.to_quaternion()
                
                # add to dynamic motion string
                dynm_string += sw.add_motion_rotation(id = f"{leaf_id}_{j}", axis=axis, angle=angle, rotation_center=rot_centre-helios_shift, nloops=frame_diff, next = f"{leaf_id}_{next_id}")
                j+= 1
            sp_string = sw.create_scenepart_obj(path, motionfilter=dynm_string, kdt_dyn_step=100)
            sceneparts += sp_string     
                    
        else:
            # add to list with static objects
            obj_paths_static.append(obj_paths_relative[i])
    
    # iterate through static obj paths and create scenepart string
    sceneparts += "\n\t\t<!--Static sceneparts-->"
    for path in obj_paths_static:
        sp_string = sw.create_scenepart_obj(path)
        sceneparts += sp_string
    
    scene = sw.build_scene(scene_id=self.scene_id, name=self.scene_name, sceneparts=[sceneparts], dyn_step=12500)
    
    # write scene to file
    with open(self.filepath, "w") as f:
        f.write(scene)


# multiply 3d coord list by matrix
def np_matmul_coords(coords, matrix, space=None):
    M = (space @ matrix @ space.inverted()
         if space else matrix).transposed()
    ones = np.ones((coords.shape[0], 1))
    coords4d = np.hstack((coords, ones))
    
    return np.dot(coords4d, M)[:,:-1]


class OT_BatchExport_DynHelios(bpy.types.Operator):
    """HELIOS - Export moving scene to HELIOS"""
    bl_idname = "helios.export"
    bl_label = "Export OBJ dynamic"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        export = scene.ExportProps
        
        # export objects (to OBJ files) 
        relative_fpaths = export_obj(export, context)

        # write the scene XML (incl. motions)
        write_scene(export, context, relative_fpaths)
        
        return {'FINISHED'}

classes = (OT_BatchExport_DynHelios,)

register, unregister = bpy.utils.register_classes_factory(classes)

if __name__ == "__main__":
    register()
