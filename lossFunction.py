import torch
import torch.nn.functional as F

def request_loss(cls_logits, request_targets):
    """
    Args
    ----
    cls_logits      : Tensor, shape (B, L*A, 2) – raw logits per anchor.
    request_targets : Tensor, shape (B,)        – 0 = benign, 1 = malicious.

    Returns
    -------
    loss            : scalar tensor – cross‑entropy between pooled logits
                      and request‑level ground truth.
    """
    # Pool over all anchors in the request – pick the most suspicious one.
    # Alternative poolers are possible (see notes below).
    request_logits = cls_logits.max(dim=1).values   # (B, 2)

    # Standard CE on the pooled logits.
    loss = F.cross_entropy(request_logits, request_targets, reduction='mean')
    return loss
