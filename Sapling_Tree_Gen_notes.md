# Blender Sapling Tree Gen Plugin

The Plugin "Sapling Tree Gen" in Blender implements the tree creation algorithm by Weber & Penn (1995).

Paper:

Blender Documentation:

## Activation

Edit -> Preferences -> Add Ons -> Add Curve: Sapling Tree Gen

## Usage

In Object Mode: Add -> Curve -> Sapling Tree Gen

A Window opens, which allows to create trees using a great number of settings.
Configurations can be stored (Geometry -> Preset (Enter Name) -> Export Preset) and loaded (Geometry -> Load Preset (Select).

Useful parameters:
- Geometry
    - Tree Scale: set size of tree
    - Shape
- Branch Radius
- Branch Splitting
- Leaves
    - Tick box "Show Leaves"
    - Leaf Down and Rot
    - Leaf Scale
    
The resulting tree object has a curve (stem; can also be converted to mesh) and a mesh (leaves).

## Post-processing

### Separating individual leaves

Select leaves. Switch to Edit mode.

Mesh -> Separate -> By Loose Parts.

Now every leaf is separated.

### Converting curve to mesh

Select tree curve -> Object -> Covert to -> Mesh from Surce/Meta/Surf/Text

### Batch export

Run `export_obj_batch.py` which exports each (mesh/curve) object in the scene to a separate file in a specified folder.
It also sets `Z` as the up-axis and `-X` as the forward-axis.