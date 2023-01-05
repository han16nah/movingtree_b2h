import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper


class ExportProps(bpy.types.PropertyGroup, ExportHelper):
    
    helios_root: bpy.props.StringProperty(name="Path to HELIOS++ root_folder", default="helios")
    sceneparts_folder: bpy.props.StringProperty(name="Name of sceneparts folder", default="tree1")
    scene_id: bpy.props.StringProperty(name="ID of the scene", default="moving_tree1")
    scene_name: bpy.props.StringProperty(name="Name of the scene", default="Tree with moving leaves")

class PT_helios(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_idname = 'helios'
    bl_label = 'HELIOS'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "scene"

    def draw(self, context):
        props = bpy.context.scene.ExportProps
        layout = self.layout
        
        # Create three rows for the input
        row = layout.row()
        row.label(text="HELIOS++ root folder")
        row.prop(props, "helios_root", text="")
        
        row = layout.row()
        row.label(text="Sceneparts folder")
        row.prop(props, "sceneparts_folder", text="")
        
        row = layout.row()
        row.label(text="Scene XML")
        row.prop(props, "filepath", text="")
        
        row = layout.row()
        row.operator("helios.export", text="Export")
        
        split = layout.split()
        col = split.column()
        col.label(text="Scene ID")
        col.prop(props, "scene_id", text="")
        
        col = split.column(align=True)
        col.label(text="scene ID")
        col.prop(props, "Scene_name", text="")


def export_button(self, context):
    self.layout.operator(
        OT_BatchExport_DynHelios.bl_idname,
        text="Batch Export OBJ dynamic",
        icon='PLUGIN')


 # Registration
def register():
    bpy.utils.register_class(PT_helios)
    bpy.utils.register_class(ExportProps)
    # register ExportProps
    bpy.types.Scene.ExportProps = bpy.props.PointerProperty(type=ExportProps)


def unregister():
    bpy.utils.unregister_class(ExportProps)
    bpy.utils.unregister_class(PT_helios)
    # $ delete ExportProps on unregister
    del(bpy.types.Scene.ExportProps)


if __name__ == "__main__":
    register()
