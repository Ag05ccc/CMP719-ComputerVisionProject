import os
import glob
import torch
import numpy as np
from PIL import Image
from utils.utils import *
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose

class KITTIOdometryDataset(Dataset):
    def __init__(self, data_dir="./data", seq_dir="sequences", transform=None, sequence_length = 11, batch_size=16):
        super(KITTIOdometryDataset, self).__init__()

        # Data directory
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # Transformation for the images
        train_transforms = Compose([
            Resize((512, 256)),
            ToTensor(),
        ])
        self.transform = transform if transform else train_transforms
        # Load image, imu and gt_pose file paths
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, seq_dir, "*/image_2/*.png")))
        self.imu_paths = sorted(glob.glob(os.path.join(data_dir, "imus/*.mat")))
        self.gt_paths = sorted(glob.glob(os.path.join(data_dir, "poses/*.txt")))

        print("self.image_paths    : ",len(self.image_paths))
        print("self.imu_paths : ",len(self.imu_paths))
        print("self.gt_paths  : ",len(self.gt_paths))

    def __len__(self):
        return len(self.image_paths)- self.sequence_length # 11  1

    def __getitem__(self, idx):

        images = []
        for i in range(idx, idx + self.sequence_length):
            # Load the images
            img_t0 = Image.open(self.image_paths[i]).convert('RGB') # ensure 3 channels
            img_t1 = Image.open(self.image_paths[i + 1]).convert('RGB') # ensure 3 channels
            # Convert images to tensors
            img1_tensor = self.transform(img_t0)
            img2_tensor = self.transform(img_t1)
            # Check that the images have the same height and width
            assert img1_tensor.shape[1:] == img2_tensor.shape[1:], "Images must have the same height and width."
            # Concatenate the two images along the channel dimension
            concat_img = torch.cat((img1_tensor, img2_tensor), dim=0).float()
            images.append(concat_img)
        concat_images = torch.stack(images) # stack images to get shape (N, C, H, W)

        # Extract the sequence and frame number from the file path
        seq_num = int(self.image_paths[idx].split(os.sep)[-3])
        frame_num = int(os.path.splitext(os.path.basename(self.image_paths[idx]))[0])
        imu_all_data = loadmat(self.imu_paths[seq_num])["imu_data_interp"]
        imu_data = imu_all_data[frame_num*10:(frame_num*10)+11]
        imu_data = np.expand_dims(imu_data, axis=1)
        imu_data = np.float32(imu_data)
        # Load ground truth pose data
        gt_poses = read_pose_from_text(self.gt_paths[seq_num])

        gt_pose = gt_poses[frame_num]
        gt_pose = matrix_to_pose(gt_pose)
        gt_pose = np.float32(gt_pose)

        h_first = torch.zeros(1024, requires_grad=False)
        h_first = (torch.zeros(2,1024),torch.zeros(2,1024))
        return concat_images, imu_data, gt_pose, h_first 
