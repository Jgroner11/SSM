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

    def forward_convolution(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        powers = self.a.pow(torch.arange(T - 1, -1, -1, dtype=x.dtype, device=x.device))
        h = (self.b * x * powers).sum(dim=1) + self.c * powers.sum()
        z = self.w * h + self.e
        return z[0] if squeeze else z


class m3(nn.Module):
    """SSM classifier
        Transition matrix for hidden state is diagonal
    """
    def __init__(self, hidden_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.a = nn.Parameter(torch.empty(hidden_size).uniform_(0.95, 0.999))
        self.b = nn.Parameter(torch.empty(hidden_size).uniform_(-0.5, 0.5))
        self.c = nn.Parameter(torch.empty(hidden_size).uniform_(-0.1, 0.1))
        self.w = nn.Parameter(torch.empty(hidden_size).uniform_(-0.5, 0.5))
        self.e = nn.Parameter(torch.empty(1).uniform_(-0.1, 0.1))

    def forward(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        for t in range(T):
            x_t = x[:, t].unsqueeze(1)
            h = self.a * h + self.b * x_t + self.c
        z = h @ self.w + self.e
        return z[0] if squeeze else z
    
    def forward_recurrent(self, x):
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        for t in range(T):
            x_t = x[:, t].unsqueeze(1)
            h = self.a * h + self.b * x_t + self.c
        z = h @ self.w + self.e
        return z[0] if squeeze else z
    
    def forward_convolution(self, x):
        # This implementation only computes the final hidden state because that is all that is needed for classification
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        if T == 0:
            h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            powers = torch.pow(
                self.a.unsqueeze(0),
                torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype).unsqueeze(1),
            )
            h = self.b * (x.unsqueeze(-1) * powers.unsqueeze(0)).sum(dim=1)
            near_one = torch.isclose(self.a, torch.ones_like(self.a))
            c_contrib = torch.where(
                near_one,
                self.c * T,
                self.c * (1 - torch.pow(self.a, T)) / (1 - self.a),
            )
            h = h + c_contrib

        z = h @ self.w + self.e
        return z[0] if squeeze else z