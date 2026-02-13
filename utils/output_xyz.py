import numpy as np


def output_xyz(filename, points):
    """Write a point cloud to an XYZ text file."""
    np.savetxt(filename, points)
