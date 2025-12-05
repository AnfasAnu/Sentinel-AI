import torch
print("torch version:", torch.__version__)
print("built with cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
