import math
import numpy as np
from scipy import ndimage

import gtsam
from gpmp2 import gpmp2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import namedtuple
from enum import Enum


class Dataset2DType(Enum):
    EmptyDataset = 'EmptyDataset'
    OneObstacleDataset = 'OneObstacleDataset'
    TwoObstaclesDataset = 'TwoObstaclesDataset'
    MultiObstacleDataset = 'MultiObstacleDataset'
    MobileMap1 = 'MobileMap1'


class ArmType(Enum):
    SimpleTwoLinksArm = 'SimpleTwoLinksArm'
    SimpleThreeLinksArm = 'SimpleThreeLinksArm'


class Dataset2D(object):
    def __init__(
        self,
        rows,
        cols,
        origin,
        cell_size,
        type=None,
    ):
        self.cols = cols
        self.rows = rows
        self.origin_x, self.origin_y = origin
        self.cell_size = cell_size
        self.map = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.type = type

    @classmethod
    def get(cls, name=None):
        if name is None or name == Dataset2DType.EmptyDataset:
            dataset = cls(300, 300, (-1, -1), 0.01, Dataset2DType.EmptyDataset)
        elif name == Dataset2DType.OneObstacleDataset:
            dataset = cls(300, 300, (-1, -1), 0.01, name)
            dataset._add_obstacle((190, 160), (60, 80))

        elif name == Dataset2DType.TwoObstaclesDataset:
            dataset = cls(300, 300, (-1, -1), 0.01, name)
            dataset._add_obstacle((200, 200), (80, 100))
            dataset._add_obstacle((160, 80), (30, 80))

        elif name == Dataset2DType.MultiObstacleDataset:
            dataset = cls(300, 400, (-20, -10), 0.1, name)
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
        elif name == Dataset2DType.MobileMap1:
            dataset = cls(500, 500, (-5, -5), 0.1, name)
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

    def get_sdf_plot(self, field, epsilon_dist=0):
        plt.figure()
        plt.set_cmap('magma')
        grid_rows = field.shape[0]
        grid_cols = field.shape[1]
        grid_corner_x = self.origin_x + (grid_cols - 1) * self.cell_size
        grid_corner_y = self.origin_y + (grid_rows - 1) * self.cell_size
        # The matlab code does some weird stuff with the origin. I'm too lazy
        # to implement that, so I'll just visualize it really quickly for now
        # TODO fix this to match
        plt.imshow(
            field,
            origin='lower',
            extent=[
                self.origin_x,
                grid_corner_x,
                self.origin_y,
                grid_corner_y
            ],
        )
        plt.colorbar()

    def get_evidence_map_plot(self):
        # TODO fix this to match matlab code better
        figure = plt.figure()
        ax = figure.add_subplot(111)
        grid_corner_x = self.origin_x + (self.cols - 1) * self.cell_size
        grid_corner_y = self.origin_y + (self.rows - 1) * self.cell_size
        if self.type == Dataset2DType.EmptyDataset:
            ax.set_xlim(self.origin_x, grid_corner_x)
            ax.set_ylim(self.origin_y, grid_corner_y)
        else:
            plt.imshow(
                (1 - self.map) * 2 + 1,
                origin='lower',
                extent=[
                    self.origin_x,
                    grid_corner_x,
                    self.origin_y,
                    grid_corner_y
                ],
            )
        return figure, ax

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


def generate_arm(arm_type, base_pose=None):
    if base_pose is None:
        base_pose = gtsam.Pose3(
            gtsam.Rot3(np.identity(3)),
            gtsam.Point3(0, 0, 0),
        )

    if arm_type == ArmType.SimpleTwoLinksArm:
        a = np.array([0.5, 0.5])
        d = np.array([0, 0])
        alpha = np.array([0, 0])
        arm = gpmp2.Arm(2, a, alpha, d, base_pose)

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

    elif arm_type == ArmType.SimpleThreeLinksArm:
        a = np.array([0.5, 0.5, 0.5])
        d = np.array([0, 0, 0])
        alpha = np.array([0, 0, 0])
        arm = gpmp2.Arm(3, a, alpha, d, base_pose)
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
            [2, -0.5, 0.0, 0.0, 0.01],
            [2, -0.4, 0.0, 0.0, 0.01],
            [2, -0.3, 0.0, 0.0, 0.01],
            [2, -0.2, 0.0, 0.0, 0.01],
            [2, -0.1, 0.0, 0.0, 0.01],
            [2, 0.0, 0.0, 0.0, 0.01],
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


def get_planar_arm_plot(fk_model, conf, color, width):
    # TODO clean this up to be more like matlab
    _position = fk_model.forwardKinematicsPosition(conf)
    # Get rid of homogeneous coordinates
    _position = _position[0:2, :]

    # TODO make this plot work with different base poses
    position = np.zeros((_position.shape[0], _position.shape[1] + 1))
    base_pose = fk_model.base_pose().translation()
    # Ignoring Z because this is a planar arm plot
    position[:, 0] = np.array([base_pose.x(), base_pose.y()])
    position[:, 1:] = _position
    line, = plt.plot(
        position[0, :],
        position[1, :],
        color=color,
        linewidth=width,
    )
    plt.plot(
        position[0, :-1],
        position[1, :-1],
        color='black',
        marker='.',
        markersize=10,
        linewidth=0)
    return line


def get_animated_planar_arm_plot(
    fig,
    ax,
    fk_model,
    confs,
    color='blue',
    width=2,
):
    positions = []
    for c in confs:
        p = fk_model.forwardKinematicsPosition(c)[0:2, :]
        position = np.zeros((p.shape[0], p.shape[1] + 1))
        base_pose = fk_model.base_pose().translation()
        # Ignoring Z because this is a planar arm plot
        position[:, 0] = np.array([base_pose.x(), base_pose.y()])
        position[:, 1:] = p
        positions.append(position)

    line, = ax.plot(
        positions[0][0, :],
        positions[0][1, :],
        color='blue',
        linewidth=2
    )

    def animate(i):
        line.set_xdata(positions[i][0, :])
        line.set_ydata(positions[i][1, :])

    anim = animation.FuncAnimation(
        fig,
        animate,
        interval=100,
        frames=len(confs),
    )
    return anim


AnimatedArmInfo = namedtuple(
    'AnimatedArmInfo',
    ['fk', 'confs', 'color', 'width'],
)


def get_animated_planar_multiarm_plot(
    fig,
    ax,
    arm_info,  # Tuples of fk_models, conf values, colors, and widths
):
    num_confs = len(arm_info[0].confs)
    pos = []
    lines = []
    for arm in arm_info:
        fk, confs, color, width = arm
        assert len(confs) == num_confs
        positions = []
        for c in confs:
            p = fk.forwardKinematicsPosition(c)[0:2, :]
            position = np.zeros((p.shape[0], p.shape[1] + 1))
            base_pose = fk.base_pose().translation()
            # Ignoring Z because this is a planar arm plot
            position[:, 0] = np.array([base_pose.x(), base_pose.y()])
            position[:, 1:] = p
            positions.append(position)
        pos.append(positions)
        line, = ax.plot(
            positions[0][0, :],
            positions[0][1, :],
            color=color,
            linewidth=width,
        )
        lines.append(line)

    def animate(time_step):
        for idx in range(len(lines)):
            line = lines[idx]
            position = pos[idx]
            line.set_xdata(pos[idx][time_step][0, :])
            line.set_ydata(pos[idx][time_step][1, :])

    anim = animation.FuncAnimation(
        fig,
        animate,
        interval=200,
        frames=num_confs,
    )
    return anim, lines


def symbol(name, value):
    return gtsam.symbol(ord(name), value)
