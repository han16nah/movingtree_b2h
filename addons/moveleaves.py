bl_info = {
    "name": "Move Leaves",
    "blender": (3, 4, 0),
    "version": (0, 1, 0),
    "author": "Hannah Weiser",
    "description": "Move Leaves of Tree generated with Sapling Tree Gen Add-on",
    "category": "Object"
}

import bpy
import numpy as np
from numpy.random import Generator
import math
from mathutils import Quaternion, Matrix, Euler

def fix_sum_random_vec(fix_sum, len, rng):
    r = rng.integers(0, fix_sum, size=(len-1))
    r = np.append(r, [0, fix_sum])
    r.sort()
    a = np.diff(r)
    
    return a


class ObjectMoveLeaves(bpy.types.Operator):
    """Object Move Leaves"""
    bl_idname = "object.move_leaves"
    bl_label = "Move Leaves"
    bl_options = {'REGISTER', 'UNDO'}

    duration: bpy.props.IntProperty(name="Duration (frames)", default=120, min=0, max=1200)
    fraction: bpy.props.FloatProperty(name="Fraction", default=0.5, min=0, max=1)
    x_mu: bpy.props.FloatProperty(name="X Angle - mean [°]", default=0, min=0, max=360)
    y_mu: bpy.props.FloatProperty(name="Y Angle - mean [°]", default=0, min=0, max=360)
    z_mu: bpy.props.FloatProperty(name="Z Angle - mean [°]", default=0, min=0, max=360)
    x_sigma: bpy.props.FloatProperty(name="X Angle - standard deviation [°]", default=15, min=0, max=360)
    y_sigma: bpy.props.FloatProperty(name="Y Angle - standard deviation [°]", default=15, min=0, max=360)
    z_sigma: bpy.props.FloatProperty(name="Z Angle - standard deviation [°]", default=15, min=0, max=360)

    def execute(self, context):
        rng = np.random.default_rng()
        scene = context.scene
        objects = scene.objects
        leaves = []
        for o in objects:
            if o.name.startswith("leaves"):
                leaves.append(o)
        moving_leaf_ids = rng.choice(len(leaves), size=int(round(self.fraction*len(leaves))), replace=False)
        moving_leaves = [leaves[i] for i in moving_leaf_ids]
        
        for leaf in moving_leaves:
            # start at frame 0
            scene.frame_set(0)
            # insert keyframe
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
            
            # get leaf base
            rot_origin = leaf.matrix_world @ leaf.data.vertices[0].co
            # set leaf base as rotation origin
            # but first: check if it already is!
            if not leaf.location == rot_origin:
                leaf.data.transform(Matrix.Translation(-rot_origin))
                leaf.location += rot_origin
            
            # durations of different leaf movements (number of frames)
            durations = fix_sum_random_vec(self.duration, math.ceil(self.duration/30), rng=rng)
            
            for d in durations:
                # move forward by duration
                scene.frame_set(scene.frame_current + d)
                # sample Euler angles
                x_angle = rng.normal(self.x_mu, self.x_sigma)
                y_angle = rng.normal(self.y_mu, self.y_sigma)
                z_angle = rng.normal(self.z_mu, self.z_sigma)
                leaf.rotation_euler = [math.radians(x_angle), math.radians(y_angle), math.radians(z_angle)]
                # insert keyframe
                leaf.keyframe_insert(data_path="rotation_euler", index=-1)
                
        
        return {'FINISHED'}


def menu_func(self, context):
    self.layout.operator(ObjectMoveLeaves.bl_idname)

# store keymaps here to access after registration
addon_keymaps = []


def register():
    bpy.utils.register_class(ObjectMoveLeaves)
    bpy.types.VIEW3D_MT_object.append(menu_func)

    # handle the keymap
    wm = bpy.context.window_manager
    # Note that in background mode (no GUI available), keyconfigs are not available either,
    # so we have to check this to avoid nasty errors in background case.
    kc = wm.keyconfigs.addon
    if kc:
        km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
        kmi = km.keymap_items.new(ObjectMoveLeaves.bl_idname, 'T', 'PRESS', ctrl=True, shift=True)
        addon_keymaps.append((km, kmi))

def unregister():
    # Note: when unregistering, it's usually good practice to do it in reverse order you registered.
    # Can avoid strange issues like keymap still referring to operators already unregistered...
    # handle the keymap
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.utils.unregister_class(ObjectMoveLeaves)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == "__main__":
    register()