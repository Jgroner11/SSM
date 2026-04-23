import torch
from torch import nn

class m1(nn):
    """SSM classifier
        Dynamical systems implementation
        Hidden state is a scalar
    """
    def __init__(self):
        self.a
        self.b
        self.c
        self.w
        self.d
        self.e

    def forward(self, x):
        T = len(x)

        h = torch.zeros(T)
        for t in range(T):
            h[t] = self.a * h[t-1] + self.b * x[t] + self.c
            # Don't need this step for classification
            # z[t] = self.w * h[t] + self.d * x[t] + self.e
        z = self.w * h[T-1] + self.d * h[T] + self.e
        return z

def loss(model : m1, X, y):
    n, T = X.shape()
    z = torch.zeros(n)
    for i in range(n):
        z[i] = m1.forward(X[i])
    p = torch.sigmoid(z)
    L = - (y * torch.log(p) + (1-y) * torch.log(1-p))
    loss = torch.mean(L)
    return loss

def train(n_iters=100):
    model = m1()
    #TODO

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(n_iters):
        optimizer.zero_grad()
        loss(m1, X, y).backward()
        optimizer.step()

    