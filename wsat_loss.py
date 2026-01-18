import torch
import torch.nn as nn


class WSATLoss(nn.Module):
    def __init__(self, delta: float = 1e-3):
        super(WSATLoss, self).__init__()
        self.delta = delta

    @staticmethod
    def _smooth_abs(x: torch.Tensor, delta: float) -> torch.Tensor:
        return torch.sqrt(torch.pow(x, 2) + delta ** 2) - delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if len(y_pred.size())>1:
            y_pred = y_pred.squeeze()
        if len(y_true.size())>1:
            y_true = y_true.squeeze()

        value_diff = y_true - y_pred
        smooth_abs_value_diff = self._smooth_abs(value_diff, self.delta)
        weight = 1.0 / (1.0 + smooth_abs_value_diff)

        angle_term1 = torch.atan2(y_true, y_pred)
        angle_term2 = torch.atan2(y_pred, y_true)
        raw_angle_diff = angle_term1 - angle_term2
        smooth_abs_angle_diff = self._smooth_abs(raw_angle_diff, self.delta)
        loss_elements = weight * torch.pow(smooth_abs_angle_diff, 2)

        return torch.mean(loss_elements)
