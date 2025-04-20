import torch
from ecgtokenizer.modeling_vqhbr import vqhbr

vqhbr_model = vqhbr()
param = torch.load("../checkpoint-100.pth", map_location=torch.device('cpu'), weights_only=False)
vqhbr_model.load_state_dict(param["model"])
