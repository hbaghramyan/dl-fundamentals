import torch
import torch.nn.functional as F

z = torch.tensor([[3.1, -2.3, 5.8], [1.1, 1.9, -8.9]])
torch.set_printoptions(precision=2, sci_mode=False)
print(F.softmax(z, dim=1))
