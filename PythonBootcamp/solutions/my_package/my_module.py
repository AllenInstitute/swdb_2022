import numpy as np

def downsample(data, n, axis=0):
    """Reduce the number of samples in an array by averaging together
    n samples at a time.
    """
    if n <= 1:
        return data
    new_len = data.shape[axis] // n
    s = list(data.shape)
    s[axis] = new_len
    s.insert(axis+1, n)
    sl = [slice(None)] * data.ndim
    sl[axis] = slice(0, new_len*n)
    d1 = data[tuple(sl)]
    d1.shape = tuple(s)
    d2 = d1.mean(axis+1)
    return d2