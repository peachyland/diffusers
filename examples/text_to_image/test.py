import torch

print(torch.__version__)
print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
