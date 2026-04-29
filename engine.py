import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import plotly.graph_objects as go
from data_gen import build_labeled_dataset, SEED
from models import m7

def accuracy(model : nn.Module, X, y):
    with torch.no_grad():
        z = model(X)
        p = torch.sigmoid(z) > 0.5
        return (p.float() == y).float().mean().item()

def train(n_iters=200, batch_size=10):
    torch.manual_seed(SEED)
    model = m7(hidden_size=10)
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
        
        # Track losses and accuracy
        train_losses.append(avg_train_loss)
        with torch.no_grad():
            test_z = model(test_X)
            test_loss = nn.functional.binary_cross_entropy_with_logits(test_z, test_y)
            test_losses.append(test_loss.item())
            train_accuracies.append(accuracy(model, X, y))
            test_accuracies.append(accuracy(model, test_X, test_y))
        
        if (iter + 1) % 10 == 0:
            print(f"Iter {iter + 1}/{n_iters} | loss={avg_train_loss:.6f}")
    
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
