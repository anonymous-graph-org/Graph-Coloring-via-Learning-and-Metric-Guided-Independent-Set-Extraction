import torch

def value_aware_bce_loss(predictions, targets, select_values, nonselect_values):
    """
    Compute Value-Aware Binary Cross-Entropy (VA-BCE) loss with enhanced weighting.
    
    Args:
        predictions (torch.Tensor): Model predictions [num_nodes, num_maps]
        targets (torch.Tensor): Ground truth labels [num_nodes]
        select_values (torch.Tensor): Select values for each node [num_nodes]
        nonselect_values (torch.Tensor): Nonselect values for each node [num_nodes]
        
    Returns:
        torch.Tensor: Value-aware loss (scalar)
    """
    # Flatten targets if needed
    targets = targets.view(-1)

    # Calculate mean values
    mean_select = torch.mean(select_values) + 1e-8
    mean_nonselect = torch.mean(nonselect_values) + 1e-8

    weights = torch.zeros_like(targets, dtype=torch.float32, device=targets.device)

    # Enhanced weighting - give more weight to high-value nodes
    # Double the weight for nodes with high select values that should be in MIS
    weights[targets == 1] = (select_values[targets == 1] / mean_select) * 2.0
    
    # Give lower weight for non-selected nodes
    weights[targets == 0] = (nonselect_values[targets == 0] / mean_nonselect) * 0.5

    losses = []
    
    for pred_idx in range(predictions.size(1)):
        pred = predictions[:, pred_idx]
        
        # Standard BCE components
        bce_pos = targets * torch.log(pred + 1e-8)
        bce_neg = (1 - targets) * torch.log(1 - pred + 1e-8)
        
        # Apply weights
        weighted_bce = weights * (bce_pos + bce_neg)
        
        # Calculate loss
        loss = -torch.mean(weighted_bce)
        losses.append(loss)
    
    # Hindsight loss - take minimum loss across all maps
    return torch.min(torch.stack(losses))