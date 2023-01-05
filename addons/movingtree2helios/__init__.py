bl_info = {
    "name": "Batch Export OBJ dynamic",
    "blender": (3, 4, 0),
    "version": (0, 1, 0),
    "author": "Hannah Weiser",
    "description": "Export moving scene to HELIOS",
    "category": "Scene"
}

import bpy
from movingtree2helios import operators, panel

modules = (operators, panel)

def register():
    for m in modules:
        m.register()


def unregister():
    for m in modules:
        m.unregister()


if __name__ == "__main__":
    register()