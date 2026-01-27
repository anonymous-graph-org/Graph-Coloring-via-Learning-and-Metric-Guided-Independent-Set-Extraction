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

class ValueAwareDeepGCN(nn.Module):
    def __init__(self, hidden_dim=32, num_layers=20, num_maps=32):
        """
        Initialize a Value-Aware GCN that takes features directly in hidden_dim.
        
        Args:
            hidden_dim (int): Dimension of hidden layers
            num_layers (int): Number of GCN layers
            num_maps (int): Number of probability maps to output
        """
        super(ValueAwareDeepGCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_maps = num_maps
        
        # Hidden GCN layers
        self.hidden_layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        
        # Output layer: transforms hidden_dim to num_maps (multiple probability maps)
        self.output_layer = nn.Linear(hidden_dim, num_maps)

    def forward(self, adjacency_matrix, degree_matrix, features):
        """
        Forward pass for the value-aware GCN.
        
        Args:
            adjacency_matrix (torch.Tensor): Adjacency matrix (N x N)
            degree_matrix (torch.Tensor): Degree matrix (N x N)
            features (torch.Tensor): Node features [N x hidden_dim] already in hidden dimension
            
        Returns:
            torch.Tensor: Multiple probability maps (N x num_maps)
        """
        # Normalize adjacency matrix: A_hat = D^(-1/2) * A * D^(-1/2)
        D_inv_sqrt = torch.diag_embed(torch.pow(degree_matrix.diag() + 1e-8, -0.5))  # Add epsilon to avoid division by zero
        A_hat = torch.matmul(torch.matmul(D_inv_sqrt, adjacency_matrix), D_inv_sqrt)
        
        # Input features are already in hidden dimension
        H = features
        
        # Apply GCN layers with residual connections
        for layer in self.hidden_layers:
            H_new = F.relu(layer(A_hat, H))
            H = H_new + H  # Add residual connection for better gradient flow
        
        # Output layer produces multiple probability maps
        output = self.output_layer(H)
        
        # Apply sigmoid to get probabilities
        return torch.sigmoid(output)  # Shape: [N x num_maps]