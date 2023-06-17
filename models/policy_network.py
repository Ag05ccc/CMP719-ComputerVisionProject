import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, temperature=5):
        prob = self.layers(x)
        decision = F.gumbel_softmax(prob, tau=temperature, hard=True, dim=-1)
        return prob, decision

# Example usage
# batch_size = 1
# input_size = 1280
# input_data = torch.randn(batch_size, input_size)

# model = PolicyNetwork()
# prob, decision = model(input_data)
# print(model)
# print(decision.shape)  # Expected output shape: (batch_size, 2)
# print(decision)