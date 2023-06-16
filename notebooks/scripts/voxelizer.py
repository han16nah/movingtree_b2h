from __future__ import print_function, division
import numpy as np
import laspy


class Voxelizer:
    def __init__(self, data, voxel_size=(1, 1, 1), method="random"):
        self.data = data
        if type(voxel_size) is not tuple:
            voxel_size = (voxel_size, voxel_size, voxel_size)
        self.voxel_size = voxel_size
        self.method = method

    def voxelize(self, origin=None):
        """
        Function to voxelize point cloud data
        Adapted from Glira (https://github.com/pglira/Point_cloud_tools_for_Matlab/
        blob/master/classes/4pointCloud/uniformSampling.m)

        :return:
        """
        # No.of points
        noPoi = self.data.shape[0]

        if origin is None:
            # Find voxel centers
            # Point with smallest coordinates
            minPoi = np.min(self.data, axis=0)

            # Rounded local origin for voxel structure
            # (voxels of different pcs have coincident voxel centers if mod(100, voxelSize) == 0)
            # localOrigin = np.floor(minPoi / 100) * 100
            localOrigin = np.floor(minPoi / 1) * 1
        else:
            localOrigin = origin

        # Find 3 - dimensional indices of voxels in which points are lying
        idxVoxel = np.array([np.floor((self.data[:, 0] - localOrigin[0]) / self.voxel_size[0]),
                             np.floor((self.data[:, 1] - localOrigin[1]) / self.voxel_size[1]),
                             np.floor((self.data[:, 2] - localOrigin[2]) / self.voxel_size[2])]).T


        # Remove multiple voxels
        idxVoxelUnique, ic = np.unique(idxVoxel, axis=0,
                                       return_inverse=True)  # ic contains "voxel index" for each point

        # Coordinates of voxel centers
        XVoxelCenter = [localOrigin[0] + self.voxel_size[0] / 2 + idxVoxelUnique[:, 0] * self.voxel_size[0],
                        localOrigin[1] + self.voxel_size[1] / 2 + idxVoxelUnique[:, 1] * self.voxel_size[1],
                        localOrigin[2] + self.voxel_size[2] / 2 + idxVoxelUnique[:, 2] * self.voxel_size[2]]
        # No.of voxel(equal to no.of selected points)
        noVoxel = len(XVoxelCenter[0])


        # Prepare list for every output voxel
        XVoxelContains = [[] for i in range(noVoxel)]
        XClosestIndex = np.full((noVoxel,), np.nan, dtype=int)

        # Select points nearest to voxel centers - --------------------------------------

        # Sort indices and points( in order to find points inside of voxels very fast in the next loop)
        idxSort = np.argsort(ic)
        ic = ic[idxSort]

        data_sorted = self.data[idxSort, :]
        idxJump, = np.nonzero(np.diff(ic))
        idxJump += 1

        # Example (3 voxel)
        # ic = [1 1 1 2 2 2 3]';
        # diff(ic) = [0 0 1 0 0 1]';
        # idxJump = [3     6]';
        #
        # idxInVoxel = [1 2 3]; for voxel 1
        # idxInVoxel = [4 5 6]; for voxel 2
        # idxInVoxel = [7];     for voxel 3

        for i in range(noVoxel):
            # Find indices of points inside of voxel(very, very fast this way)
            if i == 0:
                if i == noVoxel - 1:
                    idxInVoxel = slice(0, noPoi)
                else:
                    idxInVoxel = slice(0, idxJump[i])
            elif i == noVoxel - 1:
                idxInVoxel = slice(idxJump[i - 1], noPoi)
            else:
                idxInVoxel = slice(idxJump[i - 1], idxJump[i])

            # Fill voxel information
            XVoxelContains[i] = np.array(idxSort[idxInVoxel], dtype=int)

            # Get point closest to voxel center
            if self.method == "closest":
                distsSq = ((data_sorted[idxInVoxel, 0] - XVoxelCenter[0][i]) ** 2 +
                           (data_sorted[idxInVoxel, 1] - XVoxelCenter[1][i]) ** 2 +
                           (data_sorted[idxInVoxel, 2] - XVoxelCenter[2][i]) ** 2)
                closestIdxInVoxel = np.argmin(distsSq)
                XClosestIndex[i] = idxSort[idxInVoxel.start + closestIdxInVoxel]
            elif self.method == "random":
                XClosestIndex[i] = np.random.choice(XVoxelContains[i])

        return XVoxelCenter, XVoxelContains, idxVoxelUnique, XClosestIndex, localOrigin


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius]) 
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_result(data, voxelIdx, closestIdx, vox_size_x, vox_size_y, vox_size_z, origin):
    import matplotlib.pyplot as plt
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    mins, maxes = np.min(voxelIdx, axis=0), np.max(voxelIdx, axis=0)
    x_range = int((maxes[0] - mins[0]) / 1) + 1
    y_range = int((maxes[1] - mins[1]) / 1) + 1
    z_range = int((maxes[2] - mins[2]) / 1) + 1
    # prepare some coordinates
    voxels = np.zeros((x_range, y_range, z_range), np.bool)
    x, y, z = np.indices([v + 1 for v in voxels.shape]).astype(float)
    x = x * vox_size_x  # - vox_size_x/2
    y = y * vox_size_y  # - vox_size_y/2
    z = z * vox_size_z  # - vox_size_z/2
    # create binary representation from point list
    for voxIdx in voxelIdx:
        idxlist = (voxIdx - mins).astype(int).tolist()
        voxels[idxlist[0], idxlist[1], idxlist[2]] = True
    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[voxels] = "red"
    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, voxels, facecolors=colors, edgecolor='k', alpha=0.1, linewidth=0.1)
    # data -= np.floor(np.min(data, axis=0))
    data -= origin + mins * vox_size_x
    Xr = data[::10, 0]
    X = data[closestIdx, 0]
    Yr = data[::10, 1]
    Y = data[closestIdx, 1]
    Zr = data[::10, 2]
    Z = data[closestIdx, 2]
    ax.scatter(X, Y, Z, facecolor='b', s=1, linewidths=0)
    ax.scatter(Xr, Yr, Zr, facecolor='g', s=0.1, linewidths=0)
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    #for xb, yb, zb in zip(Xb, Yb, Zb):
    #    ax.plot([xb], [yb], [zb], 'w')
    set_axes_equal(ax)
    plt.show()


def save_subsample_pcloud(data, closestIdx, fname, delim=","):
    data_subs = data[closestIdx, :]
    np.savetxt(fname, data_subs, delimiter=delim, fmt="%.3f")


def save_subsample_pcloud_las(original_las, header, closestIdx, fname):
    new_las = laspy.LasData(header)
    new_las.points = original_las.points[closestIdx].copy()
    new_las.write(fname)


if __name__ == '__main__':
    import time
    import pandas
    import sys
    from pathlib import Path

    print("Loading file...", end='')
    t = time.time()
    infile = sys.argv[1]
    vox_size_x = vox_size_y = vox_size_z = float(sys.argv[2])
    outfile = sys.argv[3]
    if Path(infile).suffix in [".txt", ".csv", ".asc", ".xyz"]:
        data = pandas.read_csv(infile, sep='\s+', skipinitialspace=True).to_numpy(dtype=float)
    elif Path(infile).suffix in [".las", ".laz"]:
        las = laspy.read(infile)
        data = np.vstack((las.x, las.y, las.z)).transpose()
    print(" [done (%.3f s)].\nVoxelizing..." % (time.time() - t), end='')
    t = time.time()
    vox = Voxelizer(data, voxel_size=(vox_size_x, vox_size_y, vox_size_z))
    centers, idxs, voxelIdx, closestIdx = vox.voxelize()
    print(" [done (%.3f s)]." % (time.time() - t))
    print("Voxelization of %s points with a voxel size of (%s|%s|%s) resulted in %d filled voxels" % (
        data.shape[0], vox_size_x, vox_size_y, vox_size_z, closestIdx.shape[0]))
    # print(centers)
    # plot_result(data, voxelIdx, closestIdx, vox_size_x, vox_size_y, vox_size_z)
    if Path(outfile).suffix in [".txt", ".csv", ".asc", ".xyz"]:
        save_subsample_pcloud(data, closestIdx, outfile)
    elif Path(outfile).suffix in [".las", ".laz"]:
        save_subsample_pcloud_las(las, las.header, closestIdx, outfile)
