import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())



import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

print("Backend:", matplotlib.get_backend())

plt.plot([1, 2, 3], [1, 4, 9], marker="o")
plt.title("Matplotlib Test")
plt.show()