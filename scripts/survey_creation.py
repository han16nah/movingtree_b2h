#!/usr/bin/env python
# -- coding: utf-8 --

"""
Creation of H++ surveys
"""

# Imports
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import survey_creation_util as sc

# Given file paths
helios_root = "H:/helios"
scene_directory = "data/scenes/movingtrees"
survey_directory_tls = "data/surveys/movingtrees_tls"
Path(Path(helios_root) / survey_directory_tls).mkdir(parents=True, exist_ok=True)
survey_directory_uls = "data/surveys/movingtrees_uls"
Path(Path(helios_root) / survey_directory_uls).mkdir(parents=True, exist_ok=True)
scene_files = (Path(helios_root) / scene_directory).glob("tree*.xml")
scene_files = [f.as_posix() for f in scene_files]
scene_names = [Path(f).stem for f in scene_files]
survey_files_tls = [(Path(helios_root) / survey_directory_tls / f"tls_{name}.xml").as_posix() for name in scene_names]
survey_files_uls = [(Path(helios_root) / survey_directory_uls / f"uls_{name}.xml").as_posix() for name in scene_names]
plot_dir = "H:/movingtree_b2h/data/plots"

# TLS survey
############

# survey settings
n_scan_pos = 6
distance_tree = 3

# get waypoints (distributed on circle around the tree)
waypoints = sc.circle_points(distance_tree, n_scan_pos)
scanner_templ_id = "tls"

# create legs and plot the measurement plan (including FOV from each scan position)
legs = sc.create_legs_tls(waypoints, scanner_templ_id, plot=(Path(plot_dir) / "tls_survey.png").as_posix())

# scanner settings
pulse_freq = 600_000
vertical_resolution = 0.04
horizontal_resolution = 0.04
scanner_id = "riegl_vz400"

# Create and write surveys
for scene, survey in zip(scene_files, survey_files_tls):
    scene_relative = Path(scene).relative_to(helios_root).as_posix()
    scene_id = sc.get_scene_id(scene)
    survey_name = Path(survey).stem
    survey_content = sc.create_survey_tls(survey_name,
                                          scanner_templ_id,
                                          pulse_freq,
                                          vertical_resolution,
                                          horizontal_resolution,
                                          scanner_id,
                                          scene_relative,
                                          scene_id,
                                          legs)

    #with open(survey, "w") as f:
    #    f.write(survey_content)

# ULS survey
############

# we use a trajectory from a real survey
uls_traj_file1 = "../data/uls_traj1_full.txt"
uls_traj_file2 = "../data/uls_traj2_full.txt"
uls_traj1 = np.loadtxt(uls_traj_file1, delimiter=" ", comments="//")
uls_traj2 = np.loadtxt(uls_traj_file2, delimiter=" ", comments="//")
# trajectories are normalized, so let's add the flying height
flight_altitude = 15
uls_traj1[:, 2] += flight_altitude
uls_traj2[:, 2] += flight_altitude

# we do not need the full trajectory. Let's clip it by a 10 m x 10 m rectangular AOI
bbox = [-5, -5, 5, 5]
uls_traj1_c = sc.clip_by_aabb(uls_traj1, bbox)
uls_traj2_c = sc.clip_by_aabb(uls_traj2, bbox)

# plot the trajectory
figure = plt.figure(figsize=(5, 5), tight_layout=True)
plt.scatter(uls_traj1_c[:, 0], uls_traj1_c[:, 1], s=3)
plt.scatter(uls_traj2_c[:, 0], uls_traj2_c[:, 1], s=3)
plt.scatter([0], [0], marker="*", s=400, c="g")
plt.xlabel("X")
plt.ylabel("Y")
plt.text(1, 1.7, "Second grid", c="#E24A33", size=14)
plt.text(2, 4.2, "First grid", c="#348ABD", size=14)
plt.axis('square')
plt.savefig(Path(plot_dir) / "uls_survey.png", dpi=300)
# plt.show()
plt.clf()

# write clipped trajectory to file
uls_traj_file1_c = "data/trajectories/uls_traj1.txt"
uls_traj_file2_c = "data/trajectories/uls_traj2.txt"
np.savetxt(Path(helios_root) / uls_traj_file1_c, uls_traj1_c, fmt='%.6f', delimiter=',')
np.savetxt(Path(helios_root) / uls_traj_file2_c, uls_traj2_c, fmt='%.6f', delimiter=',')

# find time gaps (needed for configuration of linearly interpolated trajectories in H++)
gap_times1_s, gap_times1_e = sc.get_gaps(uls_traj1_c[:, 3])
gap_times2_s, gap_times2_e = sc.get_gaps(uls_traj2_c[:, 3])

# scanner settings
scanner_templ_id = "uls"
pulse_freq = 300_000
scan_freq = 100
scanner_id = "riegl_vux-1uav22"

slope_filter_threshold = 0.0

# first leg for trajectory file 1
legs = f'''
        <leg>
            <platformSettings 
                trajectory="{uls_traj_file1_c}" teleportToStart="true"
                tIndex="3" xIndex="0" yIndex="1" zIndex="2" rollIndex="4" pitchIndex="5" yawIndex="6"
                slopeFilterThreshold="{slope_filter_threshold}" toRadians="true" syncGPSTime="true"
                tStart="{gap_times1_s[0]}" tEnd="{gap_times1_e[0]}"
            />
            <scannerSettings template="{scanner_templ_id}" trajectoryTimeInterval_s="0.03"/>
        </leg>
       '''

# remaining legs
for i in range(1, len(gap_times1_s)):
    legs += f'''
        <leg>
            <platformSettings 
                trajectory="{uls_traj_file1_c}" teleportToStart="true"
                slopeFilterThreshold="{slope_filter_threshold}" toRadians="true" syncGPSTime="true"
                tStart="{gap_times1_s[i]}" tEnd="{gap_times1_e[i]}"
            />
            <scannerSettings template="{scanner_templ_id}" trajectoryTimeInterval_s="0.03"/>
        </leg>
       '''

# first leg for trajectory file 2
legs += f'''
        <leg>
            <platformSettings 
                trajectory="{uls_traj_file2_c}" teleportToStart="true"
                tIndex="3" xIndex="0" yIndex="1" zIndex="2" rollIndex="4" pitchIndex="5" yawIndex="6"
                slopeFilterThreshold="{slope_filter_threshold}" toRadians="true" syncGPSTime="true"
                tStart="{gap_times2_s[0]}" tEnd="{gap_times2_e[0]}"
            />
            <scannerSettings template="{scanner_templ_id}" trajectoryTimeInterval_s="0.03"/>
        </leg>
       '''

# remaining legs
for i in range(1, len(gap_times2_s)):
    legs += f'''
        <leg>
            <platformSettings 
                trajectory="{uls_traj_file2_c}" teleportToStart="true"
                slopeFilterThreshold="{slope_filter_threshold}" toRadians="true" syncGPSTime="true"
                tStart="{gap_times2_s[i]}" tEnd="{gap_times2_e[i]}"
            />
            <scannerSettings template="{scanner_templ_id}" trajectoryTimeInterval_s="0.03"/>
        </leg>
       '''

# create and write survey files
for scene, survey in zip(scene_files, survey_files_uls):
    scene_relative = Path(scene).relative_to(helios_root).as_posix()
    scene_id = sc.get_scene_id(scene)
    survey_name = Path(survey).stem
    survey_content = sc.create_survey_uls(survey_name,
                                          scanner_templ_id,
                                          pulse_freq,
                                          scan_freq,
                                          scanner_id,
                                          scene_relative,
                                          scene_id,
                                          legs)

    with open(survey, "w") as f:
        f.write(survey_content)

#with open(Path(helios_root) / survey_directory_tls / "helios_commands.txt", "w") as cf:
#    for survey in survey_files_tls:
#        cf.write(rf"run\helios {Path(survey).relative_to(helios_root)} --lasOutput --zipOutput")
#        cf.write("\n")
#
with open(Path(helios_root) / survey_directory_uls / "helios_commands.txt", "w") as cf:
    for survey in survey_files_uls:
        cf.write(rf"run\helios {Path(survey).relative_to(helios_root)} --lasOutput --zipOutput")
        cf.write("\n")
