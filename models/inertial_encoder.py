import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn

class InertialEncoder(nn.Module):
    def __init__(self):
        super(InertialEncoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv1d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 101, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).squeeze(3)
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage
# batch_size = 16
# input_channels = 6
# seq_length = 11
# input_data = torch.randn(batch_size, input_channels, seq_length)

# model = InertialEncoder()
# output = model(input_data)
# print(output.shape)  # Expected output shape: (batch_size, 256)
