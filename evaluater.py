import copy
import pprint
import matplotlib.pyplot as plt

from evo.tools import log
from evo.tools import plot
from evo.core import metrics
from evo.tools import file_interface

log.configure_logging()

# Load poses
# traj_ref = file_interface.read_kitti_poses_file("./data/poses/05.txt")
traj_ref = file_interface.read_kitti_poses_file("./05.txt")
traj_est = file_interface.read_kitti_poses_file("./results/sequences_test5_40.txt")

print(traj_ref)
print(traj_est)

# Absolute Pose Error (APE)
print("\n###########################")
print("### ABSOLUTE POSE ERROR ###")
print("###########################\n")


# max_diff = 0.01
# traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est, max_diff)
traj_est_aligned = copy.deepcopy(traj_est)
traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

fig = plt.figure()
traj_by_label = {
    "estimate (not aligned)": traj_est,
    "estimate (aligned)": traj_est_aligned,
    "reference": traj_ref
}
plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plt.show()

pose_relation = metrics.PoseRelation.translation_part
use_aligned_trajectories = True

if use_aligned_trajectories:
    data = (traj_ref, traj_est_aligned) 
else:
    data = (traj_ref, traj_est)

ape_metric = metrics.APE(pose_relation)
ape_metric.process_data(data)

ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
print(ape_stat)

ape_stats = ape_metric.get_all_statistics()
pprint.pprint(ape_stats)

plot_mode = plot.PlotMode.xyz
fig = plt.figure()
ax = plot.prepare_axis(fig, plot_mode)
plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
plot.traj_colormap(ax, traj_est_aligned if use_aligned_trajectories else traj_est, ape_metric.error, 
                   plot_mode, min_map=ape_stats["min"], max_map=ape_stats["max"])
ax.legend()
plt.show()
# --------------------------------------------------------------------------------------------------------------------------

# Relative Pose Error (RPE)
print("\n###########################")
print("### RELATIVE POSE ERROR ###")
print("###########################\n")

pose_relation = metrics.PoseRelation.rotation_angle_deg
# normal mode
delta = 1
delta_unit = metrics.Unit.frames
# all pairs mode
all_pairs = False  # activate
data = (traj_ref, traj_est)
rpe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
rpe_metric.process_data(data)

rpe_stat = rpe_metric.get_statistic(metrics.StatisticsType.rmse)
print(rpe_stat)

rpe_stats = rpe_metric.get_all_statistics()
pprint.pprint(rpe_stats)
