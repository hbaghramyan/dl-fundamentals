import random
import timeit

import torch

# a -------------------
random.seed(123)

b = 0
x = [1.2, 2.2]
w = [3.3, 4.3]

output = b
for x_j, w_j in zip(x, w):
    output += x_j * w_j

output = b

b = torch.tensor([0.])
x = torch.tensor([1.2, 2.2])
w = torch.tensor([3.3, 4.3])

output = b + x.dot(w)

print(output)

def plain_python(x, w, b):
    output = b
    for x_j, w_j in zip(x, w):
        output += x_j * w_j
    return output

b = 0.
x = [random.random() for _ in range(1000)]
w = [random.random() for _ in range(1000)]

print(timeit.timeit(lambda: plain_python(x, w, b), number=100))

def pytorch_dot(t_b, t_x, t_w):
    return t_x.dot(t_w) + t_b

t_b = torch.tensor(b)
t_x = torch.tensor(x)
t_w = torch.tensor(w)

print(timeit.timeit(lambda: pytorch_dot(t_b, t_x, t_w), number=100))

# b ----------------------

def test_py():
    b = 0.
    X = [[1.2, 2.2], [4.4, 5.5]]
    w = [3.3, 4.3]

    outputs = []
    for x in X:
        output = b
        for x_j, w_j in zip(x, w):
            output += x_j * w_j
        outputs.append(output)
    return outputs

def test_torch():

    b = torch.tensor([0.])
    X = torch.tensor(
        [[1.2, 2.2],
        [4.4, 5.5]]
    )
    w = torch.tensor([3.3, 4.3])

    return X.matmul(w) + b

print(timeit.timeit(test_py, number=10000))
print(timeit.timeit(test_torch, number=10000))

# c --------------

X = torch.rand(100, 10)
W = torch.rand(50, 10)

R = X.matmul(W.T)

print(R)

a = torch.tensor([1, 2, 3])
b = torch.tensor([4])

print(a + b)