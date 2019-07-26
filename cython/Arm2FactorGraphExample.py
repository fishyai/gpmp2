import gtsam
from gpmp2 import gpmp2
import numpy as np

import util
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Small dataset
dataset = util.Dataset2D.get('OneObstacleDataset')
rows = dataset.rows
cols = dataset.cols
cell_size = dataset.cell_size
origin_point2 = gtsam.Point2(dataset.origin_x, dataset.origin_y)

# Signed distance field
field = dataset.SDF()
sdf = gpmp2.PlanarSDF(origin_point2, cell_size, field)

# TODO update this to be more detailed
dataset.getSDFPlot(field)

# Settings
total_time_sec = 5.0
total_time_step = 10
total_check_step = 50
delta_t = total_time_sec / total_time_step
check_inter = total_check_step / total_time_step - 1

use_gp_inter = True
arm = util.generateArm('SimpleTwoLinksArm')

# GP
Qc = np.identity(2)
Qc_model = gtsam.noiseModel_Gaussian.Covariance(Qc)

# Obstacle avoidance settings
cost_sigma = 0.1
epsilon_dist = 0.1

# Prior to start/goal
pose_fix = gtsam.noiseModel_Isotropic.Sigma(2, 0.0001)
vel_fix = gtsam.noiseModel_Isotropic.Sigma(2, 0.0001)


start_conf = np.array([0, 0]).T
start_vel = np.array([0, 0]).T

end_conf = np.array([math.pi / 2, 0]).T
end_vel = np.array([0, 0]).T

avg_vel = (end_conf / total_time_step) / delta_t

# Plot param
pause_time = total_time_sec / total_time_step

# Plot start / end config
dataset.getEvidenceMapPlot()
util.getPlanarArmPlot(arm.fk_model(), start_conf, 'blue', 2)
util.getPlanarArmPlot(arm.fk_model(), end_conf, 'red', 2)

# Init optimization
graph = gtsam.NonlinearFactorGraph()
init_values = gtsam.Values()

for i in range(total_time_step + 1):

    # Chars in C++ can only be represented as integers in Python, hence ord()
    key_pos = gtsam.symbol(ord('x'), i)
    key_vel = gtsam.symbol(ord('v'), i)

    # Initialize as straight lin in configuration space
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
        graph.add(gtsam.PriorFactorVector(key_pos, end_vel, vel_fix))

    key_pos1 = gtsam.symbol(ord('x'), i - 1)
    key_pos2 = gtsam.symbol(ord('x'), i)
    key_vel1 = gtsam.symbol(ord('v'), i - 1)
    key_vel2 = gtsam.symbol(ord('v'), i)

    # A prior on the previous positions and velocities
    graph.add(
        gpmp2.GaussianProcessPriorLinear(
            key_pos1,
            key_vel1,
            key_pos2,
            key_vel2,
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
                    key_pos1,
                    key_vel1,
                    key_pos2,
                    key_vel2,
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
# use_trustregion_opt = False
#
# if use_trustregion_opt:
#     parameters = gtsam.DoglegParams()
#     parameters.setVerbosity('ERROR')
#     optimizer = gtsam.DoglegOptimizer(graph, init_values, parameters)
# else:
#     parameters = gtsam.GaussNewtonParams()
#     parameters.setVerbosity('ERROR')
#     optimizer = gtsam.GaussNewtonOptimizer(graph, init_values, parameters)

parameters = gtsam.LevenbergMarquardtParams()
parameters.setVerbosity('ERROR')
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, parameters)

optimizer.optimize()
result = optimizer.values()


fig, ax = dataset.getEvidenceMapPlot()
# plt.title('Optimized Values')
confs = [
    result.atVector(gtsam.symbol(ord('x'), i)) for i in range(total_time_step)
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

anim = animation.FuncAnimation(fig, animate, interval=100, frames=total_time_step)

plt.draw()

plt.show()
