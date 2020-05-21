from timeit import default_timer as timer
import numpy as np
import torch


def dim(x):
    if isinstance(x, np.ndarray):
        s = list(x.shape)
    else:
        s = []
        while isinstance(x, list):
            s.append(len(x))
            x = x[0]
    return s


def maxpooling(X):
    # maxpooling 2 * 2 squares in images of size m * n with stride 2
    m, n = dim(X)
    Y = [None] * (m // 2)
    for i in range(0, m - 1, 2):
        Y[int(i / 2)] = [None] * (n // 2)
        for j in range(0, n - 1, 2):
            Y[int(i / 2)][int(j / 2)] = max(
                X[i][j], X[i][j + 1], X[i + 1][j], X[i + 1][j + 1]
            )
    return np.array(Y)


m = torch.nn.MaxPool2d(2, stride=2)

info = []
for size in range(10, 251, 10):
    image = np.random.randn(size, size)
    start = timer()
    result = maxpooling(image)
    end = timer()
    info.append(end - start)
print(info)
