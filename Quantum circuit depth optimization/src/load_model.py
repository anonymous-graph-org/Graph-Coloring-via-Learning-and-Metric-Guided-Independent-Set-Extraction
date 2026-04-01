import torch

def load_model(model_path, model_class, *model_args, **model_kwargs):
    model = model_class(*model_args, **model_kwargs)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model
