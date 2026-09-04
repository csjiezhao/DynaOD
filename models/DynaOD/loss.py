import torch
import torch.nn.functional as F

def mse_loss(pred, target, mask_diagonal=True):
    """
    MSE loss with optional diagonal masking.
    Args:
        pred: [N, N] or [B, N, N]
        target: same shape as pred
        mask_diagonal: bool, whether to ignore diagonal entries
    """
    if not mask_diagonal:
        return F.mse_loss(pred, target)

    assert pred.shape == target.shape, "pred and target must have the same shape"
    device = pred.device

    if pred.dim() == 3:  # batch mode [B, N, N]
        B, N, _ = pred.shape
        mask = torch.ones((N, N), dtype=torch.bool, device=device)
        mask.fill_diagonal_(False)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
    else:  # single [N, N]
        mask = torch.ones_like(pred, dtype=torch.bool, device=device)
        mask.fill_diagonal_(False)

    diff = (pred - target)[mask]
    return torch.mean(diff ** 2)