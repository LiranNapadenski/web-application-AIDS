import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, padding=padding
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            DepthwiseSeparableConv1D(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            DepthwiseSeparableConv1D(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            DepthwiseSeparableConv1D(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Input: (batch, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)

        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2 + x1)  # Residual connection

        return x3.permute(0, 2, 1)  # back to (batch, seq_len, hidden_dim)