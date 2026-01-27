import torch
from src.model import ValueAwareDeepGCN

def load_value_aware_model(model_path, hidden_dim=32, num_layers=20, num_maps=32, device='cpu'):
    """
    Load a trained ValueAwareDeepGCN model.
    
    Args:
        model_path (str): Path to saved model state dict
        hidden_dim (int): Hidden dimension size
        num_layers (int): Number of GCN layers
        num_maps (int): Number of probability maps
        device (str): Device to load model to ('cpu' or 'cuda')
        
    Returns:
        nn.Module: Loaded ValueAwareDeepGCN model
    """
    model = ValueAwareDeepGCN(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_maps=num_maps
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model