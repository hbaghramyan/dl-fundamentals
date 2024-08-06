import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

df = pd.read_csv(
    "unit03-pytorch-training/3.6-logreg-in-pytorch/perceptron_toydata-truncated.txt",
    sep="\t",
)

X_train = df[["x1", "x2"]].values
y_train = df["label"].values

X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas


class MyDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return self.labels.shape[0]


train_ds = MyDataset(X_train, y_train)
torch.manual_seed(1)

train_loader = DataLoader(dataset=train_ds, batch_size=10, shuffle=True)

torch.manual_seed(1)
model = LogisticRegression(num_features=X_train.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

num_epochs = 20

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):

        probas = model(features)

        loss = F.binary_cross_entropy(probas, class_labels.view(probas.shape))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f"  |  Batch {batch_idx:03d}/{len(train_loader):03d}"
            f"  |  Loss: {loss:.2f}"
        )


def compute_accuracy(model, dataloarder):

    model.eval()

    correct = 0.0
    total_examples = 0

    for _, (features, class_labels) in enumerate(train_loader):

        with torch.no_grad():
            probas = model(features)

        pred = torch.where(probas > 0.5, 1, 0)
        lab = class_labels.view(pred.shape).to(pred.dtype)

        compare = lab == pred
        correct += torch.sum(compare)
        total_examples += len(compare)

    return correct / total_examples


def plot_boundary(model):

    w1 = model.linear.weight[0][0].detach()
    w2 = model.linear.weight[0][1].detach()
    b = model.linear.bias[0].detach()

    x1_min = -20
    x2_min = (-(w1 * x1_min) - b) / w2

    x1_max = 20
    x2_max = (-(w1 * x1_max) - b) / w2

    return x1_min, x1_max, x2_min, x2_max


pred = torch.where(probas > 0.5, 1, 0)

class_labels.view(pred.shape).to(pred.dtype)

train_acc = compute_accuracy(model, train_loader)

print(f"Accuracy: {train_acc*100}%")

x1_min, x1_max, x2_min, x2_max = plot_boundary(model)

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

plt.plot([x1_min, x1_max], [x2_min, x2_max], color="k")

plt.legend(loc="upper left")

plt.xlim([-5, 5])
plt.ylim([-5, 5])

plt.xlabel("Feature $x_1$", fontsize=12)
plt.ylabel("Feature $x_2$", fontsize=12)

plt.grid(visible=True)
plt.show()

x = torch.tensor([1.1, 2.1])

with torch.inference_mode():
    proba = model(x)
