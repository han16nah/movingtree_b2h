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
import math
from mathutils import Quaternion, Matrix, Euler


class ObjectMoveLeaves(bpy.types.Operator):
    """Object Move Leaves"""
    bl_idname = "object.move_leaves"
    bl_label = "Move Leaves"
    bl_options = {'REGISTER', 'UNDO'}

    fraction: bpy.props.FloatProperty(name="Fraction", default=0.5, min=0, max=1)
    x_mu: bpy.props.FloatProperty(name="X Angle - mean [°]", default=0, min=0, max=360)
    y_mu: bpy.props.FloatProperty(name="Y Angle - mean [°]", default=0, min=0, max=360)
    z_mu: bpy.props.FloatProperty(name="Z Angle - mean [°]", default=0, min=0, max=360)
    x_sigma: bpy.props.FloatProperty(name="X Angle - standard deviation [°]", default=15, min=0, max=360)
    y_sigma: bpy.props.FloatProperty(name="Y Angle - standard deviation [°]", default=15, min=0, max=360)
    z_sigma: bpy.props.FloatProperty(name="Z Angle - standard deviation [°]", default=15, min=0, max=360)

    def execute(self, context):
        scene = context.scene
        objects = scene.objects
        leaves = []
        for o in objects:
            if o.name.startswith("leaves"):
                leaves.append(o)
        moving_leaf_ids = np.random.choice(len(leaves), size=int(round(self.fraction*len(leaves))), replace=False)
        moving_leaves = [leaves[i] for i in moving_leaf_ids]
        # start at frame 0
        scene.frame_set(0)
        
        for leaf in moving_leaves:
            # insert keyframe
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
        # move forward 30 frames
        scene.frame_set(scene.frame_current + 30)
        
        for leaf in moving_leaves:
            # get leaf base
            rot_origin = leaf.matrix_world @ leaf.data.vertices[0].co
            # set leaf base as rotation origin
            # but first: check if it already is!
            if not leaf.location == rot_origin:
                leaf.data.transform(Matrix.Translation(-rot_origin))
                leaf.location += rot_origin
            
            # sample Euler angles
            x_angle = np.random.normal(self.x_mu, self.x_sigma)
            y_angle = np.random.normal(self.y_mu, self.y_sigma)
            z_angle = np.random.normal(self.z_mu, self.z_sigma)
            leaf.rotation_euler = [math.radians(x_angle), math.radians(y_angle), math.radians(z_angle)]
            # insert keyframe
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
        scene.frame_set(scene.frame_current + 30)
        
        for leaf in moving_leaves:    
            x_angle = np.random.normal(self.x_mu, self.x_sigma)
            y_angle = np.random.normal(self.y_mu, self.y_sigma)
            z_angle = np.random.normal(self.z_mu, self.z_sigma)
            leaf.rotation_mode = leaf.rotation_mode[::-1]
            leaf.rotation_euler = [math.radians(x_angle), math.radians(y_angle), math.radians(z_angle)]
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
        scene.frame_set(scene.frame_current + 30)
        
        for leaf in moving_leaves:
            
            # sample Euler angles
            x_angle = np.random.normal(self.x_mu, self.x_sigma)
            y_angle = np.random.normal(self.y_mu, self.y_sigma)
            z_angle = np.random.normal(self.z_mu, self.z_sigma)
            leaf.rotation_mode = leaf.rotation_mode[::-1]
            leaf.rotation_euler = [math.radians(x_angle), math.radians(y_angle), math.radians(z_angle)]
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
        bpy.context.scene.frame_current = bpy.context.scene.frame_current + 30
        
        for leaf in moving_leaves:   
            leaf.rotation_euler = [0, 0, 0]
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
        scene.frame_set(scene.frame_current + 30)
        
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