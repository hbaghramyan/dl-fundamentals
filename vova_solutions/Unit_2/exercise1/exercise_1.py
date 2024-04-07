import torch
import pandas as pd

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features)
        self.bias = torch.tensor(0.0)
    def forward(self, x):
        weighted_sum_z = torch.dot(x, self.weights) + self.bias
        '''
        if weighted_sum_z > 0.0:
            prediction = torch.tensor(1)
        else:
            prediction = torch.tensor(0)
        '''
        prediction = torch.where(weighted_sum_z > 0.0, 1.0, 0.0)) ## Exercise 1:
        return prediction
    def update(self, x, true_y):
        prediction = self.forward(x)
        error = true_y - prediction

        #updating
        self.bias += error
        self.weights += error * x ## Here we used tensors broadcasting feauture (which completes uncompatable vectors(matrices) to compotable size) 
    
        return error

def train_modified(model, all_x, all_y):
    error_count = -1
    i = 1
    while error_count != 0:
        error_count = 0
        for x,y in zip (all_x, all_y):
            error = model.update(x,y)
            error_count += abs(error)
        print(f"Epochs {i} errors {error_count}")
        i += 1

df = pd.read_csv("perceptron_toydata.txt", sep = '\t')

x_train = df[["0.77","-1.14"]].values
y_train = df["0"].values

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

per = Perceptron(num_features = 2)

train_modified(per, x_train, y_train)