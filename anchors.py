import torch

def generate_anchors(seq_len, anchor_sizes, stride=4):
    """
    Generate anchors for a 1D sequence.

    Args:
        seq_len (int): Length of the input sequence.
        anchor_sizes (list[int]): List of anchor sizes (heights).
        stride (int): Sliding stride.

    Returns:
        Tensor of shape (num_anchors, 2), each row is (start, end).
    """
    anchors = []
    for center in range(0, seq_len, stride):
        for size in anchor_sizes:
            start = center - size // 2
            end = center + size // 2
            start = max(start, 0)
            end = min(end, seq_len)
            if end - start > 3:
                anchors.append([start, end])
    return torch.tensor(anchors, dtype=torch.long)