import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import ValueAwareGraphDataset
from src.model import ValueAwareDeepGCN
from src.value_aware_loss import value_aware_bce_loss

def train_value_aware_model(train_data, config, logger, device='cpu'):
    logger.log(f"Training value-aware GCN model...")

    dataset = ValueAwareGraphDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ValueAwareDeepGCN(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_maps=config.NUM_MAPS
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        model.train()

        for adjacency_matrix, degree_matrix, node_features, solutions in dataloader:
            adjacency_matrix = adjacency_matrix.squeeze(0).to(device)
            degree_matrix = degree_matrix.squeeze(0).to(device)
            solutions = solutions.squeeze(0).to(device)  # squeeze batch

            # Get the number of nodes from the adjacency matrix
            num_nodes = adjacency_matrix.size(0)
            
            # Extract select and nonselect values from node_features
            select_values = node_features.squeeze(0)[:, 0].to(device)
            nonselect_values = node_features.squeeze(0)[:, 1].to(device)
            
            # Create hidden dimension features
            features = torch.zeros(num_nodes, config.HIDDEN_DIM, device=device)
            
            # Set the first two dimensions with select/nonselect values
            features[:, 0] = select_values
            features[:, 1] = nonselect_values
            
            # Set the remaining dimensions to ones
            features[:, 2:] = 1.0
            
            num_solutions = solutions.size(0)
            if num_solutions == 0:
                continue

            solution_idx = epoch % num_solutions
            solution = solutions[solution_idx]

            optimizer.zero_grad()

            output = model(adjacency_matrix, degree_matrix, features)

            loss = value_aware_bce_loss(output, solution, select_values, nonselect_values)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.log(f"Epoch {epoch+1}/{config.EPOCHS} - Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    logger.log(f"Model saved to {config.MODEL_SAVE_PATH}")

    return model