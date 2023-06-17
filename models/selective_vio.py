import torch
import torch.nn as nn
import torch.optim as optim
from .pose_network import * 
from .policy_network import * 
from .visual_encoder import * 
from .inertial_encoder import * 

class SelectiveVIO(nn.Module):
    def __init__(self):
        super(SelectiveVIO, self).__init__()
        
        self.visual_encoder = VisualEncoder()
        self.inertial_encoder = InertialEncoder()
        self.pose_network = PoseEstimationNetwork()
        self.policy_network = PolicyNetwork()
        self.batch_size = 16

        # self.visual_requires_grad(False)
        # self.inertial_requires_grad(False)
        # self.pose_requires_grad(False)        
        self.policy_requires_grad(False)
        self.grad_flag = True

    def visual_requires_grad(self, flag):
        print("visual_encoder PARAMETER SET TO REQ_GRAD_",flag)
        for param in self.visual_encoder.parameters():
            param.requires_grad = flag

    def inertial_requires_grad(self, flag):
        print("inertial_encoder PARAMETER SET TO REQ_GRAD_",flag)
        for param in self.inertial_encoder.parameters():
            param.requires_grad = flag

    def pose_requires_grad(self, flag):
        print("pose_network PARAMETER SET TO REQ_GRAD_",flag)
        for param in self.pose_network.parameters():
            param.requires_grad = flag

    def policy_requires_grad(self, flag):
        print("policy_network PARAMETER SET TO REQ_GRAD_",flag)
        for param in self.policy_network.parameters():
            param.requires_grad = flag

    def forward(self, epoch, warmup_epoch, iteration, imgs, imu, hidden=None, temperature=5):

        if epoch>warmup_epoch and self.grad_flag:
            self.grad_flag = False
            self.policy_requires_grad(True)

        batch_size = imu.shape[0]

        if hidden is None:
            hidden = (torch.zeros(batch_size,2,1024).cuda().float(), torch.zeros(batch_size,2,1024).cuda().float())
            hidden_ = hidden[0].contiguous()[:, -1, :]
        else:
            hidden_ = hidden[0].contiguous()[:, -1, :]

        # Imu Encoder
        imu_encoded = self.inertial_encoder(imu)
        
        # Policy Network
        if imu_encoded.shape[0] != hidden_.shape[0]:
            policy_input = torch.cat((imu_encoded, hidden_[:imu_encoded.shape[0]]), 1)
        else:
            policy_input = torch.cat((imu_encoded, hidden_), 1)  
        prob, decision = self.policy_network(policy_input.detach(), temperature)
        
        # Random policy for first 40 epoch
        if epoch<warmup_epoch:
            decision = (torch.randint(low=0, high=2, size=(batch_size, 2)) < 0.5).cuda().float()

        # Reshape matrix to (16, 1, 1, 1)
        decision_filter = decision[:, 1].view(batch_size, 1, 1, 1)
        if iteration>1:
            imgs_filtered = imgs * (decision_filter)
        else:
            imgs_filtered = imgs

        # Visual Encoder
        img_encoded = self.visual_encoder(imgs_filtered)

        # Pose Network
        if img_encoded.shape[0] != hidden[0].shape[0]:
            pose_hidden = (hidden[0][:img_encoded.shape[0]], hidden[1][:img_encoded.shape[0]])
        else:
            pose_hidden = hidden
        pose_input = torch.unsqueeze(torch.cat((img_encoded, imu_encoded),-1), dim=1)
        pose_estimation, hc = self.pose_network(pose_input, pose_hidden)

        return pose_estimation, hc, prob, decision

# Example usage
# batch_size = 16
# input_channels = 6
# height = 512
# width = 256
# seq_length = 11
# h_size = 1024
# img = torch.randn(batch_size, input_channels, height, width)
# imu = torch.randn(batch_size, input_channels, seq_length)
# hidden = torch.randn(batch_size, h_size)

# model = SelectiveVIO()
# result_pose, result_hidden_state = model(img, imu, hidden)
# print(result_pose.shape)  # Expected output shape: (batch_size, 6)
# print(result_hidden_state.shape)  # Expected output shape: (batch_size, 1024)
