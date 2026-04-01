import torch

def load_model(model_path, model_class, *model_args, **model_kwargs):
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
