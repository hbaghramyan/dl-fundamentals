import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time

start_time = time.time()

train_dataset = datasets.MNIST(
    root=r"./mnist",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

test_dataset = datasets.MNIST(
    root=r"./mnist", train=False, transform=transforms.ToTensor()
)
torch.manual_seed(1)

train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
)

train_counter = Counter()

for images, labels in train_loader:
    train_counter.update(labels.tolist())

print("\nTraining label distribution:")
print(sorted(train_counter.items()))

val_counter = Counter()
for images, labels in val_loader:
    val_counter.update(labels.tolist())

print("\nValidation labels distribution:")
print(sorted(val_counter.items()))

test_counter = Counter()
for images, labels in test_loader:
    test_counter.update(labels.tolist())

print("\nTest labels distribution;")
print(sorted(test_counter.items()))

majority_class = test_counter.most_common(1)[0]
print("Majority class:", majority_class[0])

baseline_acc = majority_class[1] / sum(test_counter.values())
print("Accuracy when predicting the majority class:")
print(f"{baseline_acc:.2f} ({baseline_acc*100:.2f}%)")

for images, labels in train_loader:
    break

# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training images:")
# plt.imshow(
#     np.transpose(
#         torchvision.utils.make_grid(
#             images[:64], padding=1, pad_value=1.0, normalize=True
#         ),
#         (1, 2, 0),
#     ),
# )


# plt.show()


class PyTorchMLP(nn.Module):
    def __init__(self, num_features: int, num_classes: int) -> None:
        """
        Args:
            num_features (int): number of features
            num_classes (int): number of classes
        """

        super().__init__()

        self.all_layers = nn.Sequential(
            # 1st hidden layer
            nn.Linear(in_features=num_features, out_features=50),
            nn.ReLU(),
            # 2nd hidden layer
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),
            # output layer
            nn.Linear(in_features=25, out_features=num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits


def compute_accuracy(model: PyTorchMLP, dataloader: DataLoader) -> float:
    """
    Computes the accuracy of the given model on the provided dataset.

    Args:
        model (PyTorchMLP): The neural network model to evaluate.
        dataloader (DataLoader): The dataloader that provides batches of input features
            and corresponding labels.
    Returns:
        accuracy (float): The accuracy of the model, computed as the number of correct
            predictions divided by the total number of examples.
    """
    model.eval()

    correct = 0.0
    total_examples = 0

    for features, labels in dataloader:
        features = features.to("mps")
        labels = labels.to("mps")

        with torch.inference_mode():
            logits = model(features)

        predictions = torch.argmax(logits, dim=-1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(labels)

    accuracy = correct / total_examples
    return accuracy


torch.manual_seed(1)
model = PyTorchMLP(num_features=784, num_classes=10)

model.to("mps")
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.05)

num_epochs = 10
loss_list = []
train_acc_list, val_acc_list = [], []
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to("mps")
        labels = labels.to("mps")
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not batch_idx % 250:
            # Logging
            print(
                f"Epoch {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.2f}"
            )

        loss_list.append(loss.item())

    train_acc = compute_accuracy(model=model, dataloader=train_loader)
    val_acc = compute_accuracy(model=model, dataloader=val_loader)
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%")
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
