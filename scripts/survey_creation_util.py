#!/usr/bin/env python
# -- coding: utf-8 --

"""
Util functions for creating the HELIOS surveys
"""

from pathlib import Path
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET

d = Path(os.getcwd()).parent
sys.path.append(str(d / "addons/movingtree2helios"))
import scene_writer as sp

mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
plt.style.use("ggplot")


def rotate_around_point(xy, degrees, origin=(0, 0)):
    """Function to rotate an array of points around a given origin"""
    # convert degree to radians
    radians = math.radians(degrees)
    x = xy[:, 0]
    y = xy[:, 1]
    offset_x, offset_y = origin
    # first, transform x and y to the origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    # transform and translate back
    qx = offset_x + cos_rad * adjusted_x - sin_rad * adjusted_y
    qy = offset_y + sin_rad * adjusted_x + cos_rad * adjusted_y

    return np.array((qx, qy)).T


def get_wp_and_fov_from_xml(xml: str):
    """Get the waypoints and field of view from a (HELIOS++) XMl containing only the legs"""
    tree = ET.fromstring("<root>\n" + xml + "</root>")
    x = []
    y = []
    head_rot_start = []
    head_rot_stop = []
    for leg in tree:
        for setting in leg:
            if setting.tag == "platformSettings":
                x.append(float(setting.attrib["x"]))
                y.append(float(setting.attrib["y"]))
            if setting.tag == "scannerSettings":
                head_rot_start.append(float(setting.attrib["headRotateStart_deg"]))
                head_rot_stop.append(float(setting.attrib["headRotateStop_deg"]))

    return x, y, head_rot_start, head_rot_stop


def plot_measurement_plan(xml_legs, object_position, path):
    """Plot the measurement plan for given legs (TLS case!)"""
    x, y, head_rot_start, head_rot_stop = get_wp_and_fov_from_xml(xml_legs)

    fig = plt.figure(tight_layout=True, figsize=(5, 5))
    # plt.axes().set_aspect('equal')
    scan_pos = plt.scatter(x, y, marker="x", color="black", label="Scan positions")
    tree_pos = plt.scatter(object_position[0], object_position[1], marker="*", color="g", label="Tree position", s=400)

    for x, y, start, stop in zip(x, y, head_rot_start, head_rot_stop):
        a = np.array([x, y])
        b = np.array([[x, y + 5]])
        b1 = rotate_around_point(b, start, origin=a)
        b2 = rotate_around_point(b, stop, origin=a)
        line_b1 = np.vstack((a, b1))
        line_b2 = np.vstack((a, b2))
        plt.plot(line_b1[:, 0], line_b1[:, 1], color="grey", alpha=0.5)
        fov_line, = plt.plot(line_b2[:, 0], line_b2[:, 1], color="grey", alpha=0.5, label="Field of view")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.axis("scaled")
    # plt.legend(handles=[scan_pos, tree_pos, fov_line], bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    # plt.legend(handles=[scan_pos, tree_pos, fov_line], loc="upper right")
    plt.text(0, -0.5, "Tree position", ha="center", color="g", size=14)
    plt.text(1.8, 3, "Scan position", ha="center", size=14)
    # plt.text(-2.5, 0, "Field of view", va="center", color="grey", size=12)
    plt.text(-1.9, 1.45, "Field of view", va="center", color="grey", size=14, rotation=30)
    plt.axis('square')
    plt.savefig(path, dpi=300)
    # plt.show()
    plt.clf()


def create_legs_tls(waypoints, scanner_template, tree_pos=(0, 0), fov_deg=90, plot=False):
    legs = ""

    for wp in waypoints:
        # from waypoint position, derive the necessary waypoints for the FOV (always 90°)
        vec = np.array(tree_pos) - wp  # vector from waypoint to tree position
        direction = -np.rad2deg(np.arctan2(vec[0], vec[1]))  # get direction angle
        # get start and stop angle
        rot_start = direction - fov_deg / 2
        rot_stop = direction + fov_deg / 2

        legs += f'''<leg>
            <platformSettings x="{wp[0]:.4f}" y="{wp[1]:.4f}" z="0" onGround="true"/>
            <scannerSettings template="{scanner_template}" headRotateStart_deg="{rot_start:.4f}" headRotateStop_deg="{rot_stop:.4f}" trajectoryTimeInterval_s="0.05"/>
        </leg>
        '''

    if plot:
        plot_measurement_plan(legs, tree_pos, plot)

    return legs


def get_scene_id(scene_path):
    tree = ET.parse(scene_path)
    root = tree.getroot()
    for child in root:
        if child.tag == "scene":
            scene_id = child.attrib["id"]
            break

    return scene_id


def create_survey_tls(survey_name,
                      scanner_template_id,
                      pulse_freq,
                      vertical_res,
                      horizontal_res,
                      scanner_id,
                      scene,
                      scene_id,
                      legs):
    tls_survey_content = f'''<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <scannerSettings id="{scanner_template_id}" active="true" pulseFreq_hz="{pulse_freq}" verticalResolution_deg="{vertical_res}" horizontalResolution_deg="{horizontal_res}"/>
        <survey name="{survey_name}" platform="data/platforms.xml#tripod" scanner="data/scanners_tls.xml#{scanner_id}" scene="{scene}#{scene_id}">
            {legs}
        </survey>
    </document>
    '''

    return tls_survey_content


def circle_points(r, n):
    """Create n points on a circle with radius r (centered at the origin)"""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.c_[x, y]


def clip_by_aabb(pc, aabb):
    """Clip a point cloud (stored in numpy array) by a given 2D axis-aligned bounding box [xmin, ymin, xmax, ymax]"""
    assert len(aabb) == 4
    x_min, y_min, x_max, y_max = aabb
    subset = pc[(pc[:, 0] > x_min) &
                (pc[:, 0] < x_max) &
                (pc[:, 1] > y_min) &
                (pc[:, 1] < y_max)]

    return subset


def get_gaps(arr):
    """get time gaps in trajectory"""
    idxs = np.where(np.diff(arr) > 1)[0]
    gap_times_start = list(zip(arr[idxs + 1], arr[idxs + 1]))
    gap_times_start = np.unique(np.array(gap_times_start).flatten())
    gap_times_start = np.insert(gap_times_start, 0, arr[0])
    gap_times_end = list(zip(arr[idxs], arr[idxs]))
    gap_times_end = np.unique(np.array(gap_times_end).flatten())
    gap_times_end = np.append(gap_times_end, arr[-1])

    return gap_times_start, gap_times_end


def create_survey_uls(survey_name,
                      scanner_template_id,
                      pulse_freq,
                      scan_freq,
                      scanner_id,
                      scene,
                      scene_id,
                      legs):
    uls_survey_content = f'''<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <scannerSettings id="{scanner_template_id}" active="true" pulseFreq_hz="{pulse_freq}" scanFreq_hz="{scan_freq}" />
        <survey name="{survey_name}" platform="interpolated" baseplatform="copter_linearpath" scanner="data/scanners_als.xml#{scanner_id}" scene="{scene}#{scene_id}">
            {legs}
        </survey>
    </document>
    '''

    return uls_survey_content
