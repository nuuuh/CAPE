import torch
from torch import nn

def get_loss(name, param):
    if name == 'mse':
        loss = nn.MSELoss()
        return loss
    elif name == 'focal':
        loss = focal_loss(param)
        return loss
    elif name == 'huber':
        loss = nn.HuberLoss(delta=param)
        return loss

class focal_loss(nn.Module):
    def __init__(self, gamma):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        
    def forward(self, preds, gt):
        
        loss = self.criterion(preds, gt)
        loss = (torch.pow(torch.abs(preds-gt), self.gamma)).mean()*loss
        return loss

def orthogonality_loss(M):
    MtM = torch.matmul(M.t(), M)
    I = torch.eye(MtM.size(0)).to(M.device)
    loss = torch.norm(MtM - I, p='fro')**2
    return loss




