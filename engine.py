import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
from data_gen import build_labeled_dataset, SEED

class m2(nn.Module):
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
        # This implementation only computes the final hidden state because that is all that is needed for classification
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        T = x.size(1)
        if T == 0:
            h = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        else:
            powers = torch.pow(self.a, torch.arange(T - 1, -1, -1, device=x.device, dtype=x.dtype))
            h = self.b * (x * powers).sum(dim=1)
            if self.c.abs().item() > 0:
                if torch.isclose(self.a, torch.ones_like(self.a)):
                    h = h + self.c * T
                else:
                    h = h + self.c * (1 - torch.pow(self.a, T)) / (1 - self.a)

        z = self.w * h + self.e
        return z[0] if squeeze else z


def accuracy(model: m2, X, y):
    with torch.no_grad():
        z = model(X)
        p = torch.sigmoid(z) > 0.5
        return (p.float() == y).float().mean().item()

def train(n_iters=200, batch_size=10):
    torch.manual_seed(SEED)
    model = m2()
    data = build_labeled_dataset()
    X = torch.Tensor(data["X"])
    y = torch.Tensor(data["y"]).float()
    
    # 90/10 train-test split
    split_idx = int(len(X) * 0.9)
    test_X = X[split_idx:]
    test_y = y[split_idx:]
    X = X[:split_idx]
    y = y[:split_idx]
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for iter in range(n_iters):
        epoch_train_loss = 0.0
        n_batches = 0
        
        # Loop over mini-batches from DataLoader
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            train_z = model(X_batch)
            train_loss = nn.functional.binary_cross_entropy_with_logits(train_z, y_batch)
            train_loss.backward()
            optimizer.step()
            
            epoch_train_loss += train_loss.item()
            n_batches += 1
        
        # Average loss over batches for this epoch
        avg_train_loss = epoch_train_loss / n_batches
        
        # Debug: print parameter values
        if (iter + 1) % 10 == 0:
            print(f"Iter {iter+1}: a={model.a.item():.6f}, b={model.b.item():.6f}, avg_loss={avg_train_loss:.4f}")
        
        # Track losses and accuracy
        train_losses.append(avg_train_loss)
        with torch.no_grad():
            test_z = model(test_X)
            test_loss = nn.functional.binary_cross_entropy_with_logits(test_z, test_y)
            test_losses.append(test_loss.item())
            train_accuracies.append(accuracy(model, X, y))
            test_accuracies.append(accuracy(model, test_X, test_y))
        
        # Print progress every 10 iterations
        if (iter + 1) % 10 == 0:
            print(
                f"Iter {iter + 1}/{n_iters} | "
                f"Avg Train Loss: {avg_train_loss:.4f} | "
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
