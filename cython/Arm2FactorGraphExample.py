import gtsam
from gpmp2 import gpmp2
import numpy as np

import util
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import argparse


def plan(plot=False):
    # Small dataset
    dataset = util.Dataset2D.get('OneObstacleDataset')
    origin_point2 = gtsam.Point2(dataset.origin_x, dataset.origin_y)

    # Signed distance field
    field = dataset.SDF()
    sdf = gpmp2.PlanarSDF(origin_point2, dataset.cell_size, field)

    # Settings
    total_time_sec = 5.0
    total_time_step = 10
    total_check_step = 50
    delta_t = total_time_sec / total_time_step
    check_inter = total_check_step / total_time_step - 1

    use_gp_inter = True
    arm = util.generate_arm('SimpleTwoLinksArm')

    # GP
    Qc = np.identity(2)
    Qc_model = gtsam.noiseModel_Gaussian.Covariance(Qc)

    # Obstacle avoidance settings
    cost_sigma = 0.1
    epsilon_dist = 0.1

    # Prior to start/goal
    pose_fix = gtsam.noiseModel_Isotropic.Sigma(2, 0.0001)
    vel_fix = gtsam.noiseModel_Isotropic.Sigma(2, 0.0001)

    start_conf = np.array([0, 0])
    start_vel = np.array([0, 0])

    end_conf = np.array([math.pi / 2, 0])
    end_vel = np.array([0, 0])

    avg_vel = (end_conf / total_time_step) / delta_t

    # Init optimization
    graph = gtsam.NonlinearFactorGraph()
    init_values = gtsam.Values()

    for i in range(total_time_step + 1):

        # Chars in C++ can only be represented as
        # integers in Python, hence ord()
        # The utils function wraps this behavior
        key_pos = util.symbol('x', i)
        key_vel = util.symbol('v', i)

        # Initialize as straight line in configuration space
        pose = \
            start_conf * (total_time_step - i) / total_time_step \
            + end_conf * i / total_time_step
        vel = avg_vel

        init_values.insert(key_pos, pose)
        init_values.insert(key_vel, vel)

        # At the beginning and end of the loop, we need to set the hard priors
        if i == 0:
            graph.add(gtsam.PriorFactorVector(key_pos, start_conf, pose_fix))
            graph.add(gtsam.PriorFactorVector(key_vel, start_vel, vel_fix))
            continue

        if i == total_time_step:
            graph.add(gtsam.PriorFactorVector(key_pos, end_conf, pose_fix))
            graph.add(gtsam.PriorFactorVector(key_vel, end_vel, vel_fix))

        previous_key_pos = util.symbol('x', i - 1)
        previous_key_vel = util.symbol('v', i - 1)

        # A prior on the previous positions and velocities
        graph.add(
            gpmp2.GaussianProcessPriorLinear(
                previous_key_pos,
                previous_key_vel,
                key_pos,
                key_vel,
                delta_t,
                Qc_model,
            )
        )

        # Cost factor
        graph.add(
            gpmp2.ObstaclePlanarSDFFactorArm(
                key_pos,
                arm,
                sdf,
                cost_sigma,
                epsilon_dist,
            )
        )

        # GP cost factor
        if use_gp_inter and check_inter > 0:
            for j in range(check_inter):
                tau = (j + 1) * (total_time_sec / total_check_step)
                graph.add(
                    gpmp2.ObstaclePlanarSDFFactorGPArm(
                        previous_key_pos,
                        previous_key_vel,
                        key_pos,
                        key_vel,
                        arm,
                        sdf,
                        cost_sigma,
                        epsilon_dist,
                        Qc_model,
                        delta_t,
                        tau,
                    )
                )

    # Optimize!
    use_trustregion_opt = False

    if use_trustregion_opt:
        parameters = gtsam.DoglegParams()
        parameters.setVerbosity('ERROR')
        optimizer = gtsam.DoglegOptimizer(graph, init_values, parameters)
    else:
        parameters = gtsam.GaussNewtonParams()
        parameters.setVerbosity('ERROR')
        optimizer = gtsam.GaussNewtonOptimizer(graph, init_values, parameters)

    optimizer.optimize()
    result = optimizer.values()

    if plot:
        # TODO update this to be more detailed
        dataset.get_sdf_plot(field)

        # Plot start / end config
        dataset.get_evidence_map_plot()
        util.get_planar_arm_plot(arm.fk_model(), start_conf, 'blue', 2)
        util.get_planar_arm_plot(arm.fk_model(), end_conf, 'red', 2)

        # The animation function is more custom, so I just put it here
        fig, ax = dataset.get_evidence_map_plot()

        confs = [
            result.atVector(gtsam.symbol(ord('x'), i))
            for i in range(total_time_step)
        ]

        fk = arm.fk_model()
        positions = []
        for c in confs:
            p = fk.forwardKinematicsPosition(c)[0:2, :]
            position = np.zeros((p.shape[0], p.shape[1] + 1))
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
            frames=total_time_step,
        )

        plt.draw()

        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Optimize a simple 2D dataset'
    )
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    plan(args.plot)
