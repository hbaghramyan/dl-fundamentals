import torch

b = torch.tensor([0.])
x = torch.tensor([1.2, 2.2])
w = torch.tensor([3.3, 4.3])

z = x.dot(w) + b ## Dot product
print(z)


#Matrix multiplication
X = torch.tensor([[1.2, 2.2],
                 [4.4, 5.5]])
z_vect = X.matmul(w) + b
print(z_vect)
