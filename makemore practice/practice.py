import torch
import matplotlib.pyplot as plt 

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27,27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['.'] = 26
itos = {i:s for s,i in stoi.items()}

g = torch.Generator().manual_seed(2147483647)

N = torch.zeros((27,27), dtype=torch.int32)

for w in words:
	chs = ['.']+list(w)+['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		N[ix1,ix2] += 1

