import torch
import torch.nn as nn
from torch.nn import functional as F


# toy example

torch.manual_seed(31415)
B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B, T, C)
x.shape