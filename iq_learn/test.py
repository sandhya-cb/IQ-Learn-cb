import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"NVIDIA Driver Visible to PyTorch: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version (Torch build): {torch.version.cuda}")
    # Try a tensor operation
    try:
        x = torch.tensor([1.0]).cuda()
        print("✅ Tensor operation on GPU successful.")
    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
else:
    print("❌ PyTorch cannot see the GPU.")
    print("This is likely because the installed PyTorch version was compiled with a CUDA toolkit")
    print("that is incompatible with your system driver (580.95 / CUDA 13.0).")