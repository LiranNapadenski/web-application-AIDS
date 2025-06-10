# proposal_net.py
import torch
import torch.nn as nn

class ProposalNet(nn.Module):
    def __init__(self, in_channels, num_anchors):
        """
        Args:
            in_channels (int): Number of input channels (embedding/feature size).
            num_anchors (int): Number of anchors per position.
        """
        super().__init__()
        self.num_anchors = num_anchors

        # Classification head: predict 2 logits (payload or not) per anchor
        self.cls_head = nn.Conv1d(in_channels, num_anchors * 2, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): (B, C, L), output from feature extractor

        Returns:
            cls_logits: (B, num_anchors * 2, L)
            reg_outputs: (B, num_anchors * 2, L)
        """
        cls_logits = self.cls_head(x)
        return cls_logits
