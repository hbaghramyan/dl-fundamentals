import random
import pdb
import sys

random.seed(123)

b = 0.
X = [[random.random() for _ in range(1000)] # 500 rows
     for i in range(500)]
w = [random.random() for _ in range(1000)]


X[10][10] = 'a'

def my_func(X, w, b):
    outputs = []
    for x in X:
        output = b
        for x_j, w_j in zip(x, w):
            try:
                output += x_j * w_j
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                pdb.post_mortem()
            # output += x_j * w_j
        outputs.append(output)
    return outputs

r = my_func(X, w, b)

print(r)