import torch
import numpy as np

m = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

np_ary = np.array([1., 2., 3.])
m2 = torch.from_numpy(np_ary)

m2 = m2.to(torch.float32)

other_m = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])

print(m)