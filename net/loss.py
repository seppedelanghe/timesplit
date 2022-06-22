import torch
import torch.nn as nn

class TimeSplitLoss(nn.Module):
    def __init__(self):
        super(TimeSplitLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.xy_lambda = 5
        self.s_lambda = 2

    def forward(self, predictions, target):
        
        # time loss
        t_pred, t_target = predictions[:, 0], target[:, 0]
        t_loss = self.mse(t_pred, t_target)

        # coordinates loss
        xy_pred, xy_target = predictions[:, 1:2], target[:, 1:2]
        xy_loss = self.mse(torch.flatten(xy_pred), torch.flatten(xy_target)) * self.xy_lambda

        # size loss
        s_pred, s_target = predictions[:, 3:4], target[:, 3:4]
        s_loss = self.mse(torch.flatten(s_pred), torch.flatten(s_target)) * self.s_lambda

        loss = t_loss + xy_loss + s_loss
        return loss
