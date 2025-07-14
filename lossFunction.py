import torch
import torch.nn.functional as F

def request_loss(cls_logits, request_targets):

    # Pool over all anchors in the request – pick the most suspicious one.
    request_logits = cls_logits.max(dim=1).values   # (B, 2)

    # Standard CE on the pooled logits.
    loss = F.cross_entropy(request_logits, request_targets, reduction='mean')
    return loss
