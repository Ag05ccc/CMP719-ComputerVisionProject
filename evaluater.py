import copy
import pprint
import matplotlib.pyplot as plt

from evo.tools import log
from evo.tools import plot
from evo.core import metrics
from evo.tools import file_interface

log.configure_logging()

# Load poses
traj_ref = file_interface.read_kitti_poses_file("./05.txt")
traj_est = file_interface.read_kitti_poses_file("./results/sequences_05_30.txt")
print(traj_ref)
print(traj_est)

# Absolute Pose Error (APE)
# The absolute pose error is a metric for investigating the global consistency of a SLAM trajectory
print("ABSOLUTE POSE ERROR")
traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

# Plot the trajectory
fig = plt.figure()
traj_by_label = {
    "estimate (not aligned)": traj_est,
    "estimate (aligned)": traj_est_aligned,
    "reference": traj_ref
}
plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plt.show()

# Configurations 
pose_relation = metrics.PoseRelation.translation_part
use_aligned_trajectories = True

# Data preparation
if use_aligned_trajectories:
    data = (traj_ref, traj_est_aligned) 
else:
    data = (traj_ref, traj_est)

# Run APE on Data
ape_metric = metrics.APE(pose_relation)
ape_metric.process_data(data)

# Get APE Statistics - Single
ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
print(ape_stat)
# Get all avalaible statistics at once in a dictionary
ape_stats = ape_metric.get_all_statistics()
pprint.pprint(ape_stats)

# Plot the trajectory with colormapping of the APE
plot_mode = plot.PlotMode.xyz
fig = plt.figure()
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
plot.traj_colormap(ax, traj_est_aligned if use_aligned_trajectories else traj_est, ape_metric.error, 
                   plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
ax.legend()
plt.show()

# -----------------------------------------------

# Relative Pose Error (RPE)
# The relative pose error is a metric for investigating the local consistency of a SLAM trajectory
print("RELATIVE POSE ERROR")

# Configurations
pose_relation = metrics.PoseRelation.rotation_angle_deg
delta = 1
delta_unit = metrics.Unit.frames
all_pairs = False  # activate

# Data Preparation
data = (traj_ref, traj_est)

# Run RPE on Data
rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
rpe_metric.process_data(data)

# Get RPE Statistics - Single
rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
print(rpe_stat)
# Get all avalaible statistics at once in a dictionary
rpe_stats = rpe_metric.get_all_statistics()
pprint.pprint(rpe_stats)

# Plot the trajectory with colormapping of the RPE 
# plot_mode = plot.PlotMode.xy
# fig = plt.figure()
# ax = plot.prepare_axis(fig, plot_mode)
# plot.traj(ax, plot_mode, traj_ref_plot, '--', "gray", "reference")
# plot.traj_colormap(ax, traj_est_plot, rpe_metric.error, plot_mode, min_map=rpe_stats["min"], max_map=rpe_stats["max"])
# ax.legend()
# plt.show()