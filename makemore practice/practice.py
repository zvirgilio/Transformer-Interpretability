import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

### create training set of bigrams
### pair input x with desired output y
xs, ys = [], []

for w in words:
	chs = ['.']+list(w)+['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		xs.append(ix1)
		ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

### Define initial weights of a neuron
### 27 neurons, each with 27 weights 
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27,27), requires_grad=True, generator=g)

### gradient descent:
for k in range(100):
	### now use 1-hot encoding to prepare data for input to a neural net
	xenc = F.one_hot(xs, num_classes = 27).float()

	### @ is matric mult. encoder
	logits = xenc @ W # predict log counts
	counts = logits.exp() # this is our new version of N
	probs = counts / counts.sum(1, keepdims=True) #softmax

	### last part of forward pass
	### determine loss (neg. log likelihood)
	### add regulazriation term, requires W close to 0 to minimize loss
	loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
	if k % 50 == 0:
		print(loss.item())

	### backwards pass
	W.grad=None
	loss.backward()

	### update W
	W.data += -50 * W.grad

print(loss.item())

### sample from neural net model
for i in range(15):
	out = []
	ix = 0
	while True:
		xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
		logits = xenc @ W 
		counts = logits.exp()
		p = counts/ counts.sum(1, keepdims = True)

		ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
		out.append(itos[ix])
		if ix == 0:
			break
	print(''.join(out))


# N = torch.zeros((27,27), dtype = torch.int32)

# nlls = torch.zeros(5)
# for i in range(5):
# 	# i-th bigram
# 	x = xs[i].item() #input char index
# 	y = ys[i].item() #label char index

# 	# print('-------')
# 	# print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
# 	# print('input to the nerual net:', x)
# 	# print('output probabilities from neural net:', probs[i])
# 	# print('label (actual next character):', y)
# 	p = probs[i,y]
# 	# print('probability assigned by the net to the correct character:', p.item())
# 	logp = torch.log(p)
# 	# print('log likelihood:', logp.item())
# 	nll = -logp
# 	# print('negative log likelihood:', nll.item())
# 	nlls[i] = nll 

# # print('=========')
# # print('average neg log lik:', nlls.mean().item())



### early versions 

# N = torch.zeros((27,27), dtype=torch.int32)

# for w in words:
# 	chs = ['.']+list(w)+['.']
# 	for ch1, ch2 in zip(chs, chs[1:]):
# 		ix1 = stoi[ch1]
# 		ix2 = stoi[ch2]
# 		N[ix1,ix2] += 1

### Display bigram frequency

# plt.figure(figsize = (16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
# 	for j in range(27):
# 		chstr = itos[i] +itos[j]
# 		plt.text(j,i,chstr, ha="center", va = "bottom", color='gray')
# 		plt.text(j,i, N[i,j].item(), ha="center", va="top", color='gray')
# plt.axis('off')
# plt.show()

### Create probability distribution for counts
### +1 smoothes the model by ensuring no bigram has prob. 0

# P = (N+1).float()
# P = P / P.sum(1, keepdims=True)

### predict 10 words
# for i in range(10):
# 	out=[]
# 	ix=0

# 	while True:
# 		p = P[ix]
# 		ix = torch.multinomial(p, num_samples = 1, replacement = True, generator=g).item()
# 		out.append(itos[ix])
# 		if ix == 0:
# 			break
# 	print(''.join(out))

### Determine model quality with negative log-likelihood

