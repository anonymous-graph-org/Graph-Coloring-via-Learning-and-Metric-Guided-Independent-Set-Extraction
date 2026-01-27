import torch

def load_model(model_path, model_class, *model_args, **model_kwargs):
    """
    Load a trained model from the specified path.

    Args:
        model_path (str): Path to the saved model.
        model_class (torch.nn.Module): The model class to load.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = model_class(*model_args, **model_kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model