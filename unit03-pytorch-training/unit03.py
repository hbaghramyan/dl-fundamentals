import torch
import torch.nn.functional as F
from torch.autograd import grad

# model parameters
w_1 = torch.tensor([0.23], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)

# inputs and target
x_1 = torch.tensor([1.23])
y = torch.tensor([1.0])

u = x_1 * w_1
z = u + b
print(z)

a = torch.sigmoid(z)
print(a)

loss = F.binary_cross_entropy_with_logits(z, y)
print(loss)

grad_L_w1 = grad(loss, w_1, retain_graph=True)
print(grad_L_w1)

grad_L_b = grad(loss, b, retain_graph=True)

loss.backward()

print(w_1.grad, b.grad)
