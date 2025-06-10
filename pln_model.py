import torch
import torch.nn as nn
import torch.nn.functional as F

from proposal_net import ProposalNet

class PLNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, anchor_sizes):
        """
        Args:
            vocab_size (int): Number of unique characters (vocab size)
            embedding_dim (int): Size of embedding vectors (k)
            max_length (int): Max input sequence length (Lmax)
            anchor_sizes (list): List of anchor sizes (e.g. [4, 8, 16])
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.max_length = max_length
        self.anchor_sizes = anchor_sizes
        self.num_anchors = len(anchor_sizes)

        # Simple feature extractor example - you can replace with your own
        # Using Conv1d to keep spatial dimension (sequence length)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ProposalNet outputs classification and regression heads
        self.proposal_net = ProposalNet(in_channels=128, num_anchors=self.num_anchors)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input token indices, shape (B, Lmax)

        Returns:
            cls_logits: (B, L * num_anchors, 2)
            reg_outputs: (B, L * num_anchors, 2)
        """
        # 1. Embed input (B, L) -> (B, L, embedding_dim)
        x_emb = self.embedding(x)

        # 2. Convert to (B, embedding_dim, L) for Conv1d
        x_emb = x_emb.permute(0, 2, 1)

        # 3. Feature extractor: (B, embedding_dim, L) -> (B, 128, L)
        features = self.feature_extractor(x_emb)

        # 4. Proposal network (classification + regression)
        cls_logits = self.proposal_net(features)

        # 5. Reshape outputs:
        B, C, L = cls_logits.shape
        # cls_logits shape (B, 2 * num_anchors, L) -> (B, L * num_anchors, 2)
        cls_logits = cls_logits.view(B, self.num_anchors, 2, L)
        cls_logits = cls_logits.permute(0, 3, 1, 2).contiguous()
        cls_logits = cls_logits.view(B, L * self.num_anchors, 2)

        return cls_logits
