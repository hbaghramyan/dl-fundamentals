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
