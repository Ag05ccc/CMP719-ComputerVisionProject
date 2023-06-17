import sys
import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append('../utils')
from utils import *
        
class VisualEncoder(nn.Module):
    def __init__(self, args=None):
        super(VisualEncoder, self).__init__()
        
        # FlowNet-S architecture used as a visual encoder - (except for the last layer)
        self.conv1   = utils.conv(in_channels=6,   out_channels=64, kernel_size=7, stride=2)
        self.conv2   = utils.conv(in_channels=64,  out_channels=128, kernel_size=5, stride=2)
        self.conv3   = utils.conv(in_channels=128, out_channels=256, kernel_size=5, stride=2)
        self.conv3_1 = utils.conv(in_channels=256, out_channels=256)
        self.conv4   = utils.conv(in_channels=256, out_channels=512, stride=2)
        self.conv4_1 = utils.conv(in_channels=512, out_channels=512)
        self.conv5   = utils.conv(in_channels=512, out_channels=512, stride=2)
        self.conv5_1 = utils.conv(in_channels=512, out_channels=512)
        self.conv6   = utils.conv(in_channels=512, out_channels=1024,stride=2)

        self.flatten = nn.Flatten(start_dim=1)
        self.visual_head = nn.Linear(in_features= 8 * 4 * 1024, out_features=512)

        # conv1: (512, 256, 3) -> (256, 128, 64)
        # conv2: (256, 128, 64) -> (128, 64, 128)
        # conv3: (128, 64, 128) -> (64, 32, 256)
        # conv3_1: (64, 32, 256) -> (64, 32, 256)
        # conv4: (64, 32, 256) -> (32, 16, 512)
        # conv4_1: (32, 16, 512) -> (32, 16, 512)
        # conv5: (32, 16, 512) -> (16, 8, 512)
        # conv5_1: (16, 8, 512) -> (16, 8, 512)
        # conv6: (16, 8, 512) -> (8, 4, 1024)

    def forward(self, x):
        # FlowNet-S forward pass
        # ...
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        # Flatten the conv image
        out_flatten = self.flatten(out_conv6)
        out_flatten = out_conv6.view(out_conv6.size(0), -1)
        
        x = self.visual_head(out_flatten)
        return x

# batch_size = 4
# channels = 6
# height = 512
# width = 256
# input_tensor = torch.randn(batch_size, channels, height, width)

# # Instantiate the VisualEncoder
# visual_encoder = VisualEncoder()

# # Perform a forward pass through the VisualEncoder
# output = visual_encoder(input_tensor)

# # Print the output tensor shape
# print("Output tensor shape:", output.shape)
# #Output tensor shape: torch.Size([batch_size, 512])