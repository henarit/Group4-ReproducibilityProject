
import os, random, time
from itertools import product

import numpy as np

try:
    import torch
    import torch.nn  as nn
    import torch.nn.functional as F
    from torch.autograd import Variable
    from torch import optim

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

except:
    pass

    import IPython

# Batches a (data, target) generator
def batched(datagen, batch_size=32):

    arr = []
    for data in datagen:
        arr.append(data)
        if len(arr) == batch_size:
            yield list(zip(*arr))
            arr = []

    if len(arr) > 0:
        yield list(zip(*arr))


# Turns an array (batch) of dictionaries into a dictionary of arrays
def stack(batch, targets=None):

    keys = batch[0].keys()
    data = {key: [] for key in keys}

    for item, key in product(batch, keys):
        data[key].append(item.get(key, None))
    return data


# Takes the masked mean of a tuple of data
def masked_mean(data, masks):

    num = sum((X*mask[:, None].float() for X, mask in zip(data, masks)))
    denom = sum((mask for mask in masks))[:, None].float()
    return num/denom

def masked_variance(data, masks):
    EX2 = masked_mean(data, masks)**2
    E2X = masked_mean((x**2 for x in data), masks)
    return E2X - EX2


#Measures the elapsed time from the last call
def elapsed(times=[time.time()]):
    times.append(time.time())
    return times[-1] - times[-2]




