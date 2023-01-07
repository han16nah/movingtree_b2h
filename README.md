# movingtree_b2h

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This is part of the code for my Master thesis project, where I will investigate the influence of vegetation movement (e.g., leaves shaking) during
high-resolution laser scanning acquisition on the resulting point clouds and the information derived from it.

In a first step, I am creating synthetic animated trees with leaves which undergo rigid motions. The add-on [movingleaves.py](https://github.com/han16nah/movingtree_b2h/blob/main/addons/moveleaves.py)
is a first prototype to do so. As a basis, it requires a tree object, as generated by the Add-on [Sapling Tree Gen](https://docs.blender.org/manual/en/latest/addons/add_curve/sapling.html). See [Sapling_Tree_Gen_notes.md](https://github.com/han16nah/movingtree_b2h/blob/main/Sapling_Tree_Gen_notes.md) for more info.

Next, the tree animation is exported to a HELIOS++ scene. A first prototype is implemented with the Add-on [movingtree2helios](https://github.com/han16nah/movingtree_b2h/tree/main/addons/movingtree2helios).
Each OBJ file is exported and a scene file is written which includes the rigid motion sequences for each moving leaf.


<!-- GETTING STARTED -->
## Getting Started

The repository contains two blender add-ons, as well Sapling tree presets. 

### Prerequisites

- [Blender](https://www.blender.org/) (version 3.4.0, should also work on 2.8.0)
- [HELIOS++](https://github.com/3dgeo-heidelberg/helios)
- Clone or download this repository

### Installation

The [Move Leaves](https://github.com/han16nah/movingtree_b2h/blob/main/addons/moveleaves.py) add-on is a single-file add-on. 
To install the add-on in Blender, go to Edit -> Preferences... -> Add-ons. Here, click on "Install..." and navigate to moveleaves.py in the file browser. The installed add-on now appears in the menu. Activate the add-on by clicking on the checkmark.

The [movingtree2helios](https://github.com/han16nah/movingtree_b2h/tree/main/addons/movingtree2helios) add-on contains several modules. Prior to installing, zip the folder *movingtree2helios*. Then install as described above, but selecting the zip-folder in the file browser.


<!-- USAGE EXAMPLES -->
## Usage

To use the add-ons, we first need a tree model. Activate the [https://docs.blender.org/manual/en/latest/addons/add_curve/sapling.html] add-on, which is distributed with Blender. Under Edit -> Preferences... -> Add-ons, search for "Sapling" and activate the Blugin "Add Curve: Sapling Tree Gen".
Open the Sapling menu by typing `Shift + A` in Object mode, then selecting Curve -> Sapling Tree Gen. Configure your tree or choose from a preset. See [Sapling_Tree_Gen_notes.md](https://github.com/han16nah/movingtree_b2h/blob/main/Sapling_Tree_Gen_notes.md), the [Sapling Tree Gen documentation](https://docs.blender.org/manual/en/latest/addons/add_curve/sapling.html) or video instructions online for guidance. Make sure to enable "Show Leaves" in the "Leaves" menu.

Before adding the leaf animation, make sure to
- convert the tree from curve to mesh (Select the tree curve, in Object mode select Object -> Convert -> Mesh)
- separate the leaves (Select the leaves, switch to Edit mode, hit `P` and select `By Loose Parts`)

After installation of the *moveleaves* add-on, you can animate leaf movement by Choosing Object -> Move Leaves or using the Shortcut `Shift + Ctrl + T`. You can set the fraction of leaves which should be moving as well the mean and standard deviation of a normal distribution for the three Euler angles, from which angles will be randomly sampled. More options for configuring the movement will be implemented. 

For exporting the animation to HELIOS++ format, we are using the *movingtree2helios* add-on.
After installation, you find a tab "HELIOS" in the [Scene Properties](https://docs.blender.org/manual/en/latest/scene_layout/scene/properties.html). Here, you can select your HELIOS++ root folder, a sceneparts folder, the location of the scene XML to write and set a scene ID and name. Click export to export the animation.

![grafik](https://user-images.githubusercontent.com/41050948/210868699-f8bb5fab-a3ed-4dfb-a02a-896c5d4aa707.png)

Lastly, write a survey file to virtually scan the moving tree, e.g. with terrestrial laser scanning (TLS) from multiple scan positions. Execute the survey with HELIOS++. 

<!-- LICENSE -->
## License

This project is licensed under the MIT License.
See [LICENSE.md](https://github.com/han16nah/movingtree_b2h/blob/main/LICENSE.md).



<!-- CONTACT -->
## Contact

Hannah Weiser - h.weiser@uni-heidelberg.de

Project Link: [https://github.com/han16nah/movingtree_b2h](https://github.com/han16nah/movingtree_b2h)



<!-- ACKNOWLEDGMENTS -->

