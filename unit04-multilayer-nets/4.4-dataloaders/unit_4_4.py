import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torchvision.utils as vutils
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None) -> None:
        """
        A custom dataset class for loading images and their corresponding
        labels from a CSV file and an image directory.

        Attributes:
            img_dir (str): images directory
            transform (callable, optional): A function/transform to apply to the images.
            img_names (pd.Series): A series of image file paths from the DataFrame.
            labels (pd.Series): A series of labels corresponding to the images.
        """

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # based on DataFrame columns
        self.img_names = df["filepath"]
        self.labels = df["label"]

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves the image and label at the specified index.

        Args:
            index (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing:
                - img (PIL.Image or transformed image): The image at the given index,
                potentially transformed
                - label (any): The label corresponding to the iage at the given index.
        Raises:
            FileNotFoundError: If the image file is not found in the specified directory.
        """
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]

        return img, label

    def __len__(self) -> int:
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of samples in the dataset, equivalent
            to the number of labels.
        """
        return self.labels.shape[0]


def viz_batch_images(batch):

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(batch[0][:64], padding=2, normalize=True), (1, 2, 0)
        )
    )

    plt.show()


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop((28, 28)),
            transforms.ToTensor(),
            # normalize images to [-1, 1] range
            transforms.Normalize((0.5,), (0.5)),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize(32),
            transforms.CenterCrop((28, 28)),
            transforms.ToTensor(),
            # normalize images to [-1, 1] range
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
}

if __name__ == "__main__":

    train_dataset = MyDataset(
        csv_path=r"mnist-pngs/new_train.csv",
        img_dir=r"mnist-pngs/",
        transform=data_transforms["train"],
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2
    )

    val_dataset = MyDataset(
        csv_path=r"mnist-pngs/new_val.csv",
        img_dir=r"mnist-pngs/",
        transform=data_transforms["test"],
    )

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=32, shuffle=False, num_workers=2
    )

    test_dataset = MyDataset(
        csv_path=r"mnist-pngs/test.csv",
        img_dir=r"mnist-pngs/",
        transform=data_transforms["test"],
    )

    test_loader = DataLoader(
        dataset=test_dataset, shuffle=False, batch_size=32, num_workers=2
    )

    num_epochs = 1
    for epoch in range(num_epochs):

        for batch_idx, (x, y) in enumerate(train_loader):
            time.sleep(1)
            if batch_idx >= 3:
                break
            print(" Batch index:", batch_idx, end="")
            print(" | Batch size:", y.shape[0], end="")
            print(" | x shape:", x.shape, end="")
            print(" | y shape:", y.shape)

    print("Labels from current batch:", y)

    batch = next(iter(train_loader))
    viz_batch_images(batch[0])
