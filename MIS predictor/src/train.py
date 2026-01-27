import torch
import torch.optim as optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from src.dataset import GraphDataset
from src.model import DeepGCN

def hindsight_loss(predictions, solution):
    """
    Compute hindsight loss: min_m ℓ(li, fm(Gi; θ))
    
    Args:
        predictions: Tensor of shape [num_nodes, M] where M is number of output maps
        all_solutions: Tensor of shape [num_solutions, num_nodes] containing all possible solutions
        
    Returns:
        Minimum loss across all solution-prediction pairs
    """
    losses = []
    bce_loss = BCELoss(reduction='mean')
    
    # For each possible solution
    
    solution = solution.view(-1, 1)  # Shape: [num_nodes, 1]
    # Compute loss against each prediction map
    for pred_idx in range(predictions.size(1)):
        pred = predictions[:, pred_idx].view(-1, 1)  # Shape: [num_nodes, 1]
        loss = bce_loss(pred, solution)
        losses.append(loss)
    
    # Return minimum loss
    return torch.min(torch.stack(losses))

def train_model(graphs, labels_dict, hidden_dim, num_layers, epochs, learning_rate, save_path):
    print("Training...")

    dataset = GraphDataset(graphs, labels_dict)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Modified DeepGCN will now output num_maps probability maps
    model = DeepGCN(hidden_dim=hidden_dim, num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        for adjacency_matrix, degree_matrix, all_solutions in dataloader:
            k = all_solutions[0].shape[0]
            if k==0:
                continue
            num_nodes = adjacency_matrix.size(1)
            features = torch.ones(num_nodes, hidden_dim)
            optimizer.zero_grad()
            
            # Model now outputs multiple probability maps
            output = model(adjacency_matrix[0], degree_matrix[0], features)  # Shape: [num_nodes, num_maps]
            # choose one soln from all_solns[0]

            
            # Compute hindsight loss
            loss = hindsight_loss(output, all_solutions[0][epoch%k])
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")