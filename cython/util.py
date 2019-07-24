import math
import numpy as np
from scipy import ndimage

import gtsam
from gpmp2 import gpmp2
import matplotlib.pyplot as plt


class Dataset2D(object):
    def __init__(
        self,
        rows,
        cols,
        origin,
        cell_size,
    ):
        self.cols = cols
        self.rows = rows
        self.origin_x, self.origin_y = origin
        self.cell_size = cell_size
        self.map = np.zeros((self.rows, self.cols), dtype=np.float64)

    @classmethod
    def get(cls, name):
        if name == 'OneObstacleDataset':
            dataset = cls(300, 300, (-1, -1), 0.01)
            dataset._add_obstacle((190, 160), (60, 80))

        elif name == 'TwoObstaclesDataset':
            dataset = cls(300, 300, (-1, -1), 0.01)
            dataset._add_obstacle((200, 200), (80, 100))
            dataset._add_obstacle((160, 80), (30, 80))

        elif name == 'MultiObstacleDataset':
            dataset = cls(300, 400, (-20, -10), 0.1)
            dataset._add_obstacle(
                dataset._get_center(12, 10),
                dataset._get_dim(5, 7),
            )
            dataset._add_obstacle(
                dataset._get_center(-7, 10),
                dataset._get_dim(10, 7),
            )
            dataset._add_obstacle(
                dataset._get_center(0, -5),
                dataset._get_dim(10, 5),
            )
        elif name == 'MobileMap1':
            dataset = cls(500, 500, (-5, -5), 0.1)
            dataset._add_obstacle(
                dataset._get_center(0, 0),
                dataset._get_dim(1, 5),
            )
            dataset._add_obstacle(
                dataset._get_center(0, 4.5),
                dataset._get_dim(10, 1),
            )
            dataset._add_obstacle(
                dataset._get_center(0, -4.5),
                dataset._get_dim(10, 1),
            )
            dataset._add_obstacle(
                dataset._get_center(4.5, 0),
                dataset._get_dim(1, 10),
            )
            dataset._add_obstacle(
                dataset._get_center(-4.5, 0),
                dataset._get_dim(1, 10),
            )
        else:
            raise Exception('No such dataset exists')

        return dataset

    # TODO this can definitely be optimized to use less memory
    def SDF(self):
        m = self.map > 0.75
        inv_m = 1 - m

        # Unlike bwdist in matlab, this function measures distance to closest 0
        dist = ndimage.distance_transform_edt(inv_m)
        inv_dist = ndimage.distance_transform_edt(m)

        if 0 not in dist or 0 not in inv_dist:
            # There are no obstancles in the field, so we cap infinity as in
            # the MATLAB code
            return np.ones((self.rows, self.cols), dtype=np.float64) * 1000

        sdf = dist - inv_dist
        sdf *= self.cell_size
        sdf.astype(np.float64)

        return sdf

    def getSDFPlot(self, field, epsilon_dist=0):
        plt.figure()
        grid_rows = field.shape[0]
        grid_cols = field.shape[1]
        grid_corner_x = self.origin_x + (grid_cols - 1) * self.cell_size
        grid_corner_y = self.origin_y + (grid_rows - 1) * self.cell_size
        # The matlab code does some weird stuff with the origin. I'm too lazy
        # to implement that, so I'll just visualize it really quickly for now
        # TODO fix this to match
        plt.imshow(field)
        plt.colorbar()

    def getEvidenceMapPlot(self):
        # TODO fix this to match matlab code better
        figure = plt.figure()
        ax = figure.add_subplot(111)
        grid_corner_x = self.origin_x + (self.cols - 1) * self.cell_size
        grid_corner_y = self.origin_y + (self.rows - 1) * self.cell_size
        plt.imshow((1 - self.map) * 2 + 1)
        # ax.axis('equal')
        # ax.set(
        #     xlim=(
        #         self.origin_x - self.cell_size / 2,
        #         grid_corner_x + self.cell_size / 2
        #     ),
        #     ylim=(
        #         self.origin_y - self.cell_size / 2,
        #         grid_corner_y + self.cell_size / 2
        #     )
        # )
        plt.colorbar()

    def _get_center(self, x, y):
        return (
            (y - self.origin_y) / self.cell_size,
            (x - self.origin_x) / self.cell_size,
        )

    def _get_dim(self, w, h):
        return (h / self.cell_size, w / self.cell_size)

    def _add_obstacle(
        self,
        position,
        size,
    ):
        half_row = int(math.floor((size[0] - 1) / 2))
        half_col = int(math.floor((size[1] - 1) / 2))
        row, col = position
        self.map[
            row - half_row:row + half_row,
            col - half_col:col + half_col
        ] = 1


def generateArm(arm_type, base_pose=None):
    if base_pose is None:
        base_pose = gtsam.Pose3(
            gtsam.Rot3(np.identity(3)),
            gtsam.Point3(0, 0, 0),
        )

    if arm_type == 'SimpleTwoLinksArm':
        a = np.array([0.5, 0.5])
        d = np.array([0, 0])
        alpha = np.array([0, 0])
        arm = gpmp2.Arm(2, a, alpha, d)

        spheres_data = np.array([
            [0, -0.5, 0.0, 0.0, 0.01],
            [0, -0.4, 0.0, 0.0, 0.01],
            [0, -0.3, 0.0, 0.0, 0.01],
            [0, -0.2, 0.0, 0.0, 0.01],
            [0, -0.1, 0.0, 0.0, 0.01],
            [1, -0.5, 0.0, 0.0, 0.01],
            [1, -0.4, 0.0, 0.0, 0.01],
            [1, -0.3, 0.0, 0.0, 0.01],
            [1, -0.2, 0.0, 0.0, 0.01],
            [1, -0.1, 0.0, 0.0, 0.01],
            [1, 0.0, 0.0, 0.0, 0.01],
        ])

        sphere_vec = gpmp2.BodySphereVector()
        for row in range(len(spheres_data)):
            sphere = gpmp2.BodySphere(
                spheres_data[row, 0],
                spheres_data[row, 4],
                gtsam.Point3(spheres_data[row, 1:3]),
            )
            sphere_vec.push_back(sphere)

    else:
        raise Exception('No such arm exists')

    return gpmp2.ArmModel(arm, sphere_vec)


def getPlanarArmPlot(arm, conf, color, width):
    # TODO clean this up to be more like matlab
    position = arm.forwardKinematicsPosition(conf)
    return position
