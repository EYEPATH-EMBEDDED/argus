import torch
sd = torch.load("first_best_model.pth", map_location="cpu")
print([k for k in sd.keys() if k.startswith(("conv", "fc", "head"))])