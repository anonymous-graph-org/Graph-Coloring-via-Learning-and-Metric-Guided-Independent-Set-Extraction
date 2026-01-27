import torch

def load_model(model_path, model_class, *model_args, **model_kwargs):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        model_class (torch.nn.Module): The model class to load.
        *model_args: Arguments to pass to model_class constructor.
        **model_kwargs: Keyword arguments to pass to model_class constructor.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Create model instance
    model = model_class(*model_args, **model_kwargs)
    
    # Load state dictionary
    try:
        # Try loading with specified map_location
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        # Fallback to default loading
        model.load_state_dict(torch.load(model_path))
    
    # Set to evaluation mode
    model.eval()
    
    return model