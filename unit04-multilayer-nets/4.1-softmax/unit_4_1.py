import torch
import torch.nn.functional as F

y = torch.tensor([0, 2, 2, 1])

net_inputs = torch.tensor(
    [[1.5, 0.1, -0.4], [0.5, 0.7, 2.1], [-2.1, 1.1, 0.8], [1.1, 2.5, -1.2]]
)
print(F.cross_entropy(net_inputs, y))
# activations = torch.softmax(net_inputs, dim=1)
# z = torch.tensor([[3.1, -2.3, 5.8], [1.1, 1.9, -8.9]])
# torch.set_printoptions(precision=2, sci_mode=False)
# print(F.softmax(z, dim=1))


def manual_cross_entopy(net_inputs, y):
    activations = torch.softmax(net_inputs, dim=1)
    y_onehot = F.one_hot(y)
    train_losses = -torch.sum(y_onehot * torch.log(activations), dim=1)
    avg_loss = torch.mean(train_losses)

    return avg_loss


val = manual_cross_entopy(net_inputs, y)
print(val)
