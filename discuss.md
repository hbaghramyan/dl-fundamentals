05/08/2024
1. 

class MyDataset(Dataset):

    def __init__(self, X, y):
        super().__init__() - THERE IS NO CONSTRUCTOR IN DATASET
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)

2. plt.legend(loc="upper left") instead of plt.legend(loc=2)

3. you can avoid model.train() because there are no layers that perform diffently at training and evaluation stages (like dropout and BatchNorm)

4. you do not need model = model.eval()

5. plot_boundary


26/08/24

1. Best coding practice - do not specify variable you don't use


all_x = []
for x, _ in train_loader:
    all_x.append(x)
    
train_std = torch.concat(all_x).std(dim=0)
train_mean = torch.concat(all_x).mean(dim=0)

2. explain why the formula has this concrete form

![alt text](image.png)


$H(p, q) = -\sum_i p_i \log q_i = -y \log \hat{y} - (1 - y) \log(1 - \hat{y}).$


https://en.wikipedia.org/wiki/Cross-entropy

https://en.wikipedia.org/wiki/Entropy_(information_theory)

3. Do not remember the functions and parameters - read the documentation

4. plt.imshow(np.transpose(torchvision.utils.make_grid(images[:64], padding=4, pad_value=0.5, normalize=True), (1, 2, 0)))

padding - specifies the amount of space (in pixels) to insert between the images in the grid.
pad_value - specifies the pixel value used to fill the padding spaces between the images.
normalize (bool, optional): If True, shift the image to the range (0, 1),
by the min and max values specified by ``value_range``. Default: ``False``.