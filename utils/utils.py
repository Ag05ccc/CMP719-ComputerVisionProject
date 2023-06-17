import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def conv(in_channels, out_channels, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1,inplace=True)
        )

def read_pose_from_text(path):
    with open(path) as f:
        lines = f.readlines()
        num_poses = len(lines)
        poses_abs = np.empty((num_poses, 4, 4), dtype=np.float32)
        poses_rel = np.empty((num_poses - 1, 3, 4), dtype=np.float32)
        prev_pose = np.vstack((np.fromstring(lines[0], sep=' ').reshape(3, 4), [0, 0, 0, 1]))
        poses_abs[0] = prev_pose

        for i in range(1, num_poses):
            curr_pose = np.vstack((np.fromstring(lines[i], sep=' ').reshape(3, 4), [0, 0, 0, 1]))
            poses_abs[i] = curr_pose
            prev_pose = curr_pose
    return poses_abs


def matrix_to_pose(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 transformation matrix.")

    # Extract translation vector (tx, ty, tz)
    translation = matrix[:3, 3]

    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Compute Euler angles (rx, ry, rz) using the XYZ convention
    rx = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    ry = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    rz = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Combine translation and rotation into the 6-DoF pose
    pose = np.array([translation[0], translation[1], translation[2], rx, ry, rz])
    
    return pose

def make_log_file(info):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    file_index = 1
    log_file = f"./logs/log-{file_index}.txt"
    while os.path.isfile(log_file):
        file_index += 1
        log_file = f"./logs/log-{file_index}.txt"

    with open(log_file, "w") as file:
        # file.write(info.replace(",\s+", ","))
        file.write(info)

def calculate_errors(pose_estimation, decision, gt_poses):
    """
    pose_estimation.shape :  torch.Size([16, 6])
    decision.shape        :  torch.Size([16, 2])
    gt_poses.shape        :  torch.Size([16, 6])
    """
    # Calculate RMSE
    t_rmse, r_rmse = compute_rmse(pose_estimation, gt_poses)
    return t_rmse, r_rmse

def compute_rmse(pose_estimation, gt_poses):
    assert(len(gt_poses) == len(pose_estimation))
    pose_estimation = pose_estimation.cpu().detach().numpy()
    gt_poses = gt_poses.cpu().detach().numpy()
    t_rmse = np.sqrt(np.mean(np.sum((pose_estimation[:, 3:] - gt_poses[:, 3:])**2, -1)))
    r_rmse = np.sqrt(np.mean(np.sum((pose_estimation[:, :3] - gt_poses[:, :3])**2, -1)))
    return t_rmse, r_rmse

def convert_6dof_to_transformation_matrix(input_tensor):
    assert input_tensor[1].shape[-1] == 6
    transformation_matrices = []
    for i, pose in enumerate(input_tensor):
        tx, ty, tz, rx, ry, rz = pose.tolist()
        rotation_matrix = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        translation_vector = np.array([tx, ty, tz])
        transformation_matrix = np.hstack([rotation_matrix, translation_vector.reshape(3,1)])
        transformation_matrices.append(transformation_matrix.flatten())
    return torch.tensor(transformation_matrices)

def write_to_txt(pose_estimation, file_name="result_estimations.txt"):
    assert pose_estimation[1].shape[0] == 6  # now allowing any number of rows
    pose_estimation = convert_6dof_to_transformation_matrix(pose_estimation)
    with open(file_name, 'a') as f:
        for row in pose_estimation:
            row_str = ' '.join(map(str, row.tolist()))
            f.write(row_str + '\n')

def save_policy_usage(policy_list, file_name="policy_usage.txt"):
    with open(file_name, 'a') as f:
        for row in policy_list:
            row_str = ' '.join(map(str, row.tolist()))
            f.write(row_str + '\n')
