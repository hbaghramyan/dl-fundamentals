import torch


## Tensors like C arrays

array = torch.Tensor([[1,2,3],
                     [4,5,6]]) ## In this case will return [2,3]


## Checking data type of tensor


print(array.shape) ## Returns dimension of tensor
print(array.ndim) ## Number of dimension of tensor
print(array.dtype) ## Type of elements of tensor

## Changing data type of tensor
array = array.to(torch.float32)


## From numpy to tensor

import numpy as np
np_ary = np.array([1,2,3])
m2 = torch.from_numpy(np_ary) ## more efficent for memory than torch.tensor(np_arr)


####
print(array.device) ## Prints where the tensor is located

####


print(array.view(3,2))  ## Change matrix to (3,2) dimensional
print(array.view(-1,6)) ## When we put -1, then the corresponding dimension calculated automatically
                        ## in this case it will output [1,2,3,4,5,6]
##

print(array.T) ## Transposing of matrix 'array'
