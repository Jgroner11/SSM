import torch
from torch import nn

class m1(nn.Module):
    """SSM classifier
        Dynamical systems implementation
        Hidden state is a scalar
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.empty(1).uniform_(0.95, 0.999))
        self.b = nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5))
        self.c = nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1))
        self.w = nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5))
        self.e = nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1))

    def forward(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        h = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        for t in range(T):
            h = self.a * h + self.b * x[:, t] + self.c
        z = self.w * h + self.e
        return z[0] if squeeze else z
    
class m2(nn.Module):
    """SSM classifier
        Dynamical systems implementation
        Hidden state is a scalar
        convolution for more efficient training
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.empty(1).uniform_(0.95, 0.999))
        self.b = nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5))
        self.c = nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1))
        self.w = nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5))
        self.e = nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1))

    def forward(self, x, mode = "convolution"):
        if mode in ("conv", "convolution"):
            return self.forward_convolution(x)
        elif mode == "recurrent":
            return self.forward_recurrent(x)
        else:
            raise ValueError("mode must be 'conv', 'convolution', or 'recurrent'")

    def forward_recurrent(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        h = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        for t in range(T):
            h = self.a * h + self.b * x[:, t] + self.c
        z = self.w * h + self.e
        return z[0] if squeeze else z