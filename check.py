import torch

print("PyTorch version:", torch.__version__)  # Should return 2.4.1
print("CUDA version:", torch.version.cuda)  # Should return 11.8
print("cuDNN version:", torch.backends.cudnn.version())  # Should return 8600
print("CUDA available:", torch.cuda.is_available())  # Should return True
print("GPU name:", torch.cuda.get_device_name(0))  # Should show your GPU