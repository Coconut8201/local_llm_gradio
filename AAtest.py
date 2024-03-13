import torch

# 檢查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")

# 顯示 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 顯示 CUDA 版本
print("CUDA version:", torch.version.cuda)