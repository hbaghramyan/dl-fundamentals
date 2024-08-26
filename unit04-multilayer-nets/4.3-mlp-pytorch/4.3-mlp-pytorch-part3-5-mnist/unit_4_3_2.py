import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torchvision

train_dataset = datasets.MNIST(
    root=r"./mnist", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = datasets.MNIST(
    root=r"./mnist", train=False, transform=transforms.ToTensor()
)

torch.manual_seed(1)

train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training images:")
plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(
            images[:64], padding=1, pad_value=1.0, normalize=True
        ),
        (1, 2, 0),
    ),
)

plt.show()
