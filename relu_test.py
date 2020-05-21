from timeit import default_timer as timer
import numpy as np
import torch


def relu(x):
    return np.vectorize(lambda a: (a >= 0) * a)(x)


m = torch.nn.ReLU()

info = []
for size in range(10, 301, 10):
    image = torch.randn((size, size))
    start = timer()
    result = relu(image)
    end = timer()
    info.append(end - start)
print(info)
