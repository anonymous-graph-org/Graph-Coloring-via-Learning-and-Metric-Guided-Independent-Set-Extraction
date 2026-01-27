import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.theta0 = nn.Linear(in_dim, out_dim, bias=False)
        self.theta1 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, A_hat, H):
        """
        Forward pass for a single GCN layer.
        Args:
            A_hat (torch.Tensor): Normalized adjacency matrix (N x N).
            H (torch.Tensor): Feature matrix from the previous layer (N x Cl).
        Returns:
            torch.Tensor: Updated feature matrix for the next layer (N x Cl+1).
        """
        propagated_H = torch.matmul(A_hat, H)
        H_next = torch.matmul(H, self.theta0.weight) + torch.matmul(propagated_H, self.theta1.weight)
        return H_next

class DeepGCN(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(DeepGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers


        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        

        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, adjacency_matrix, degree_matrix, features):
        """
        Forward pass for the GCN.
        Args:
            adjacency_matrix (torch.Tensor): Adjacency matrix (N x N).
            degree_matrix (torch.Tensor): Degree matrix (N x N).
            features (torch.Tensor): Input feature matrix (N x hidden_dim).
        Returns:
            torch.Tensor: Predicted output (N x hidden_dim).
        """
        # Normalize adjacency matrix: A_hat = D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag_embed(torch.pow(degree_matrix.diag(), -0.5))
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0  
        A_hat = torch.matmul(torch.matmul(D_inv_sqrt, adjacency_matrix), D_inv_sqrt)

        H = features

        # Hidden layers with ReLU
        for layer in self.hidden_layers:
            H = F.relu(layer(A_hat, H))
 
        H = self.output_layer(H)


        return torch.sigmoid(H)  # Final output (N x 1)
