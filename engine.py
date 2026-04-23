import torch
from torch import nn
import plotly.graph_objects as go
from data_gen import build_labeled_dataset, SEED

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
        T = len(x)

        h = torch.tensor(0.0)
        for t in range(T):
            h = self.a * h + self.b * x[t] + self.c
        z = self.w * h + self.e
        return z

def loss(model : m1, X, y):
    n, T = X.shape
    z = torch.zeros(n)
    for i in range(n):
        z[i] = model(X[i])
    p = torch.sigmoid(z)
    L = - (y * torch.log(p) + (1-y) * torch.log(1-p))
    loss = torch.mean(L)
    return loss


def accuracy(model: m1, X, y):
    n = len(X)
    correct = 0
    for i in range(n):
        p = torch.sigmoid(model(X[i])) > 0.5
        correct += (p.float() == y[i]).item()
    return correct / n

def train(n_iters=100):
    torch.manual_seed(SEED)
    model = m1()
    data = build_labeled_dataset()
    X = torch.Tensor(data["X"])
    y = torch.Tensor(data["y"]).float()
    
    # 90/10 train-test split
    split_idx = int(len(X) * 0.9)
    test_X = X[split_idx:]
    test_y = y[split_idx:]
    X = X[:split_idx]
    y = y[:split_idx]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for iter in range(n_iters):
        optimizer.zero_grad()
        train_loss = loss(model, X, y)
        train_loss.backward()
        optimizer.step()
        
        # Debug: print parameter values
        if (iter + 1) % 10 == 0:
            print(f"Iter {iter+1}: a={model.a.item():.6f}, b={model.b.item():.6f}, loss={train_loss.item():.4f}")
        
        # Track losses and accuracy
        train_losses.append(train_loss.item())
        with torch.no_grad():
            test_loss = loss(model, test_X, test_y)
            test_losses.append(test_loss.item())
            train_accuracies.append(accuracy(model, X, y))
            test_accuracies.append(accuracy(model, test_X, test_y))
        
        # Print progress every 10 iterations
        if (iter + 1) % 10 == 0:
            print(
                f"Iter {iter + 1}/{n_iters} | "
                f"Train Loss: {train_loss.item():.4f} | "
                f"Test Loss: {test_loss.item():.4f} | "
                f"Train Acc: {train_accuracies[-1]:.3f} | "
                f"Test Acc: {test_accuracies[-1]:.3f}"
            )
    
    # Plot train and test accuracy
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_accuracies, mode='lines', name='Train Accuracy'))
    fig.add_trace(go.Scatter(y=test_accuracies, mode='lines', name='Test Accuracy'))
    fig.update_layout(
        title='Training Accuracy',
        xaxis_title='Iteration',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified'
    )
    fig.show()

if __name__ == "__main__":
    train()