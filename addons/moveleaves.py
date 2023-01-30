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
from mathutils import Quaternion, Matrix, Euler, Vector
import warnings


def planophile(angle):
    return np.abs(2/np.pi*(1+np.cos(2*angle)))


def erectophile(angle):
    return np.abs(2/np.pi*(1-np.cos(2*angle)))


def plagiophile(angle):
    return np.abs(2/np.pi*(1-np.cos(4*angle)))


def extremophile(angle):
    return np.abs(2/np.pi*(1+np.cos(4*angle)))


def spherical(angle):
    return np.abs(np.sin(angle))


def uniform(angle):
    return np.abs(2/np.pi+0*angle)


LADS = {"planophile": planophile,
        "erectophile": erectophile,
        "plagiophile": plagiophile,
        "extremophile": extremophile,
        "spherical": spherical,
        "uniform": uniform}


def sample_angles_by_dist(num_leaves, distribution):
    # get specified LAD
    lad = LADS[distribution]
    # sample angles in 1° steps
    angles = np.linspace(0, math.pi/2, 91)
    angles_deg = angles*180/np.pi
    
    # get probabilities per angle
    vals = np.empty(shape=len(angles))
    for j, a in enumerate(angles):
        vals[j] = lad(a)
    # normalize values
    probs = vals/sum(vals)
    
    # sample from the angle distribution
    rng = np.random.default_rng()
    leaf_angles = rng.choice(angles_deg, size=num_leaves, p=probs)
    
    return leaf_angles


def validate_lad(lad):
    if lad in LADS.keys():
        pass
    else:
        warnings.warn(f"Leaf angle distribution '{lad}' does not exist. Setting to default 'planophile'")
        lad = ""
    
    return lad


class ObjectMoveLeaves(bpy.types.Operator):
    """Object Move Leaves"""
    bl_idname = "object.move_leaves"
    bl_label = "Move Leaves"
    bl_options = {'REGISTER', 'UNDO'}

    seed: bpy.props.IntProperty(name="Random seed", default=42)
    duration: bpy.props.IntProperty(name="Duration (frames)", default=120, min=0, max=1200)
    lad: bpy.props.StringProperty(name="Leaf angle distribution", default="")
    fraction: bpy.props.FloatProperty(name="Fraction", default=1.0, min=0, max=1)
    frequency: bpy.props.FloatProperty(name="Oscillation frequency [hz]", default=6, min=0, max=20)
    x_mu: bpy.props.FloatProperty(name="X Angle - mean [°]", default=0, min=0, max=360)
    y_mu: bpy.props.FloatProperty(name="Y Angle - mean [°]", default=0, min=0, max=360)
    z_mu: bpy.props.FloatProperty(name="Z Angle - mean [°]", default=0, min=0, max=360)
    x_sigma: bpy.props.FloatProperty(name="X Angle - standard deviation [°]", default=8, min=0, max=360)
    y_sigma: bpy.props.FloatProperty(name="Y Angle - standard deviation [°]", default=2, min=0, max=360)
    z_sigma: bpy.props.FloatProperty(name="Z Angle - standard deviation [°]", default=2, min=0, max=360)

    def execute(self, context):
        rng = np.random.default_rng(seed=self.seed)
        scene = context.scene
        fps = scene.render.fps
        objects = scene.objects
        leaves = []
        for o in objects:
            if o.name.startswith("leaves") or o.name.startswith("Leaves"):
                leaves.append(o)
        moving_leaf_ids = rng.choice(len(leaves), size=int(round(self.fraction*len(leaves))), replace=False)
        moving_leaves = [leaves[i] for i in moving_leaf_ids]
        
        if not self.lad == "":
            lad = validate_lad(self.lad)
            print(lad)
            leaf_angles = sample_angles_by_dist(len(moving_leaves), lad)
        z = Vector((0, 0, 1))
        
        # duration in number of frames from frame rate and oscillation frequency
        dur_per_mov = int(round(fps / self.frequency))
        #todo: add random noise to durations (per leaf)?
        durations = np.empty(shape=int(np.ceil(self.duration/dur_per_mov)))
        durations.fill(dur_per_mov)
        durations[-1] = int(self.duration - sum(durations[:-1]))
        
        for i, leaf in enumerate(moving_leaves):
            # start at frame 0
            scene.frame_set(0)
            
            # get leaf base (= rotation origin)
            rot_origin = leaf.matrix_world @ leaf.data.vertices[0].co
            # set leaf base as rotation origin
            # but first: check if it already is!
            if not leaf.location == rot_origin:
                leaf.data.transform(Matrix.Translation(-rot_origin))
                leaf.location += rot_origin
            
            # get normal vector
            # any face from the mesh will do, as leaves are all planar, so let's just get the first
            normal_vec = leaf.data.polygons[0].normal.to_4d()
            normal_vec.w = 0
            normal_vec = (leaf.matrix_world @ normal_vec).to_3d()
            
            # if no leaf angle distribution is set, keep the leaves as they are.
            # Else, set leaf orientations according to sampled leaf angles and then apply rotations
            if not self.lad == "":
                # we want to rotate aligned to the normal vector, therefore:
                q = normal_vec.rotation_difference(z).to_euler()
                # rotate by leaf angle
                # get random angle theta
                theta = rng.uniform(0, 2*np.pi)
                # get unit vector pointing in that direction
                axis = [math.cos(theta), math.sin(theta), 0]
                # use unit vector as axis in axis-angle representation
                rot = Matrix.Rotation(np.deg2rad(leaf_angles[i]), 4, axis)
                q.rotate(rot)
                leaf.rotation_euler = q
            
            # apply rotations
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.transform_apply(rotation = True, location = False, scale = False)
            bpy.ops.object.select_all(action='DESELECT')
            
            # insert keyframe
            leaf.keyframe_insert(data_path="rotation_euler", index=-1)
            
            for d in durations:
                # move forward by duration
                scene.frame_set(scene.frame_current + int(d))
                # sample Euler angles
                x_angle = rng.normal(self.x_mu, self.x_sigma)
                y_angle = rng.normal(self.y_mu, self.y_sigma)
                z_angle = rng.normal(self.z_mu, self.z_sigma)
                
                # rotation
                eul = Euler((math.radians(x_angle), math.radians(y_angle), math.radians(z_angle)), 'XYZ')
                #eul.rotate(q)
                
                leaf.rotation_euler = eul
                # insert keyframe
                leaf.keyframe_insert(data_path="rotation_euler", index=-1)
                
        scene.frame_set(0)
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