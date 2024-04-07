import torch
import pandas as pd

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features)
        self.bias = torch.tensor(0.0)
    def forward(self, x):
        weighted_sum_matrix = x.matmul(self.weights) + b # Here x must be input data matrix
        prediction = torch.where(weighted_sum_matrix > 0.0, 1.0, 0.0) ## Returns vector of predictions (see torch.where() docs)
        return prediction
    ## How to implement update() function in this case?

