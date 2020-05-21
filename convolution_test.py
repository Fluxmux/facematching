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


def convolution(X, W):
    m, n = dim(X)
    s = len(W)  # s * s filter W
    s2 = (s - 1) // 2
    Y = [None] * m
    for i in range(m):
        Y[i] = [None] * n
        for j in range(n):
            t = 0
            ix = i - s2
            for di in range(s):
                if 0 <= ix < m:
                    jx = j - s2
                    for dj in range(s):
                        if 0 <= jx < n:
                            t += X[ix][jx] * W[di][dj]
                        jx += 1
                ix += 1
            Y[i][j] = t
    return Y


m = torch.nn.Conv2d(1, 1, 3, padding=1)

info = []
for size in range(10, 251, 10):
    image = torch.randn(1, 1, size, size)
    start = timer()
    result = m(image)
    end = timer()
    info.append(end - start)
print(info)
