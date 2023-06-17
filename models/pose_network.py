import torch
import torch.nn as nn
import torch.optim as optim

class PoseEstimationNetwork(nn.Module):
    def __init__(self):
        super(PoseEstimationNetwork, self).__init__()
        self.lstm = nn.LSTM(768, 1024, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )
        
    def forward(self, x, prev=None):
        # x shape: (batch_size(16), seq_length(1), input_size(768)) when bactch_first=True
        if prev == None:
            lstm_out, hc = self.lstm(x)
        else:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
            lstm_out, hc = self.lstm(x, prev)
        
        # lstm_out shape: (batch_size, seq_length, lstm_hidden_size)
        # Only last lstm hidden state required
        last_hidden_state = lstm_out[:, -1, :]
        
        # last_hidden_state shape: (batch_size, lstm_hidden_size)
        pose_estimation = self.mlp(last_hidden_state)
        
        # Returned hc[0] hc[1] shape     : torch.Size([batch_size, 2, 1024])
        # Returned pose_estimation shape : torch.Size([batch_size, 1, 6]) (raw) / torch.Size([batch_size, 6]) (custom)
        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())

        return pose_estimation, hc

# Example usage
# input_size = 768
# batch_size = 16
# seq_length = 1
# input_data = torch.randn(batch_size, seq_length, input_size)
# print(input_data.shape)
# model = PoseEstimationNetwork()
# pose_output, hidden = model(input_data)
# print(pose_output.shape)  # Expected output shape: (batch_size, 6)
# print(hidden.shape)
