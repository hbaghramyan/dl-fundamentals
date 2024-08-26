import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class PyTorchMLP(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.all_layers = nn.Sequential(
            # first hidden layer
            nn.Linear(num_features, 25),
            nn.ReLU(),
            # second hidden layuer
            nn.Linear(25, 15),
            nn.ReLU(),
            # output layer
            nn.Linear(15, num_classes),
        )

    def forward(self, x):
        logits = self.all_layers(x)

        return logits


class MyDataset(Dataset):
    def __init__(self, X, y):

        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.int64)

    def __getitem__(self, index):

        x = self.features[index]
        y = self.labels[index]

        return x, y

    def __len__(self):
        return self.labels.shape[0]


def compute_accuracy(model, dataloader):

    model.eval()

    correct = 0.0
    total_examples = 0

    for _, (features, labels) in enumerate(dataloader):

        with torch.inference_mode():  # basically the same as torch.no_grad
            logits = model(features)

        predictions = torch.argmax(logits, dim=-1)

        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


df = pd.read_csv(
    r"unit04-multilayer-nets/4.3-mlp-pytorch/4.3-mlp-pytorch-part1-2-xor/xor.csv"
)

X = df[["x1", "x2"]].values
y = df["class label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=1,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=1,
    stratify=y_train,
)


train_ds = MyDataset(X_train, y_train)
val_ds = MyDataset(X_val, y_val)
test_ds = MyDataset(X_test, y_test)

train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=32,
    shuffle=True,
)

val_dataloader = DataLoader(
    dataset=val_ds,
    batch_size=32,
    shuffle=False,
)

test_dataloader = DataLoader(dataset=test_ds, shuffle=False, batch_size=32)

torch.manual_seed(1)
model = PyTorchMLP(num_features=2, num_classes=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)  # Stochastic gradient descent

num_epochs = 10

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_dataloader):

        logits = model(features)
        loss = F.cross_entropy(logits, labels)  # loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        print(
            f"Epoch: {epoch+1:03d}/{batch_idx:03d}"
            f" | Batch {batch_idx:03d}/{len(train_dataloader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

    train_acc = compute_accuracy(model, train_dataloader)
    val_acc = compute_accuracy(model, val_dataloader)
    print(f"Train Acc {train_acc * 100:.2f}% | Val Acc {val_acc * 100:.2f}%")


train_acc = compute_accuracy(model, train_dataloader)
val_acc = compute_accuracy(model, val_dataloader)
test_acc = compute_accuracy(model, test_dataloader)

print(f"Train Acc {train_acc*100:.2f}%")
print(f"Val Acc {val_acc*100:.2f}%")
print(f"Test Acc {test_acc*100:.2f}%")

from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ("D", "^", "x", "s", "v")
    colors = ("C0", "C1", "C2", "C3", "C4")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    tensor = torch.tensor(np.array([xx1.ravel(), xx2.ravel()]).T).float()
    logits = classifier.forward(tensor)
    Z = np.argmax(logits.detach().numpy(), axis=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            color=cmap(idx),
            # edgecolor='black',
            marker=markers[idx],
            label=cl,
        )


plot_decision_regions(X_train, y_train, classifier=model)

plt.plot(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    marker="D",
    markersize=10,
    linestyle="",
    label="Class 0",
)

plt.plot(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    marker="^",
    markersize=13,
    linestyle="",
    label="Class 1",
)

plt.legend(loc=2)

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid()
plt.show()
