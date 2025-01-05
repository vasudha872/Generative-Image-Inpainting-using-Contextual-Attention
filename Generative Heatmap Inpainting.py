import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import os

class HeatmapGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HeatmapGNN, self).__init__()
        self.gnn1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gnn2 = GATConv(4 * hidden_dim, hidden_dim, heads=4, concat=True)
        self.fc = nn.Linear(4 * hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        return self.fc(x)

def construct_graph(heatmap, mask):
    n_rows, n_cols = heatmap.shape
    edges = []
    node_features = heatmap.flatten().reshape(-1, 1)
    node_features[mask.flatten()] = 0

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < n_rows and 0 <= nj < n_cols:
                    nidx = ni * n_cols + nj
                    edges.append([idx, nidx])

    edge_index = torch.tensor(edges, dtype=torch.long).T
    return Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index)

def plot_heatmap(data, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

def train_model(model, data_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            optimizer.zero_grad()
            out = model(data)

            target = data.x.clone()
            mask = target == 0
            loss = criterion(out[~mask], target[~mask])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")
    return model

def process_heatmap(file_path, mask_ratio=0.2, epochs=10):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    heatmap = np.load(file_path)
    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be a 2D NumPy array.")

    mask = np.random.rand(*heatmap.shape) < mask_ratio

    plot_heatmap(heatmap, "Original Heatmap")
    plot_heatmap(mask.astype(float), "Mask (Missing Regions)")

    graph_data = construct_graph(heatmap, mask)
    loader = DataLoader([graph_data], batch_size=1)

    model = HeatmapGNN(input_dim=1, hidden_dim=16, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    trained_model = train_model(model, loader, optimizer, criterion, epochs)

    trained_model.eval()
    with torch.no_grad():
        predicted = trained_model(graph_data).reshape(heatmap.shape).numpy()
        reconstructed_heatmap = heatmap.copy()
        reconstructed_heatmap[mask] = predicted[mask]

    plot_heatmap(reconstructed_heatmap, "Reconstructed Heatmap")
    return reconstructed_heatmap

if __name__ == "__main__":
    file_path = "/Users/gurunishalsaravanan/PycharmProjects/ML Final Project/00001.npy"  # Example file name
    reconstructed = process_heatmap(file_path, mask_ratio=0.2, epochs=10)