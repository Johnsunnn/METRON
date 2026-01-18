import pandas as pd
import torch
import math
import configs
from wsat_loss import WSATLoss


def angular_difference_pytorch(BA: torch.Tensor, CA: torch.Tensor) -> torch.Tensor:
    try:
        pi_val = torch.pi
    except AttributeError:
        pi_val = torch.tensor(math.pi, dtype=BA.dtype, device=BA.device)
    ang_diff = torch.atan2(BA, CA) - (pi_val / 4.0)
    return torch.mean(ang_diff)


def calculate_mae(preds, targets):
    return torch.mean(torch.abs(preds - targets))


def calculate_mse(preds, targets):
    return torch.mean((preds - targets) ** 2)


def calculate_rmse(preds, targets):
    return torch.sqrt(calculate_mse(preds, targets))


def calculate_r_squared(preds, targets):
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    if ss_tot < configs.METRIC_EPSILON:
        return torch.tensor(0.0, device=preds.device)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_pcc(preds, targets):
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    if preds_flat.shape[0] < 2:
        return torch.tensor(0.0, device=preds.device)
    std_vx = torch.std(preds_flat)
    std_vy = torch.std(targets_flat)
    if std_vx < configs.METRIC_EPSILON or std_vy < configs.METRIC_EPSILON:
        return torch.tensor(0.0, device=preds.device)
    combined = torch.stack((preds_flat, targets_flat), dim=0)
    pcc_matrix = torch.corrcoef(combined)
    pcc_val = pcc_matrix[0, 1]
    if torch.isnan(pcc_val):
        return torch.tensor(0.0, device=preds.device)
    return pcc_val


def calculate_mape(preds, targets):
    mask = torch.abs(targets) > configs.METRIC_EPSILON
    if not torch.any(mask):
        return torch.tensor(float('inf'), device=preds.device)
    mape = torch.mean(torch.abs((targets[mask] - preds[mask]) / (targets[mask]))) * 100
    return mape


def get_metrics(predictions, targets):
    preds_flat = predictions.squeeze().detach().cpu()
    targets_flat = targets.squeeze().detach().cpu()

    preds_flat = preds_flat.float()
    targets_flat = targets_flat.float()

    metrics = {}
    metrics['ang_diff'] = angular_difference_pytorch(preds_flat, targets_flat).item()
    wsatl = WSATLoss()
    metrics['wsatl'] = wsatl(preds_flat, targets_flat).item()
    metrics['mae'] = calculate_mae(preds_flat, targets_flat).item()
    metrics['mse'] = calculate_mse(preds_flat, targets_flat).item()
    metrics['rmse'] = calculate_rmse(preds_flat, targets_flat).item()
    metrics['mape'] = calculate_mape(preds_flat, targets_flat).item()
    metrics['pcc'] = calculate_pcc(preds_flat, targets_flat).item()
    metrics['r2'] = calculate_r_squared(preds_flat, targets_flat).item()

    return metrics