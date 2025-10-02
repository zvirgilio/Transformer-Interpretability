import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
words = open('names.txt', 'r').read().splitlines()

g = torch.Generator().manual_seed(2147483647)

#build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

### classes keep a self.out to be able to visualize and keep track of activation

class Linear:

	def __init__(self, fan_in, fan_out, bias=True):
		self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
		self.bias = torch.zeros(fan_out) if bias else None

	def __call__(self, x):
		self.out = x @ self.weight
		if self.bias is not None:
			self.out += self.bias 
		return self.out 

	def parameters(self):
		return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1d:

	def __init__(self, dim, eps = 1e-5, momentum = 0.1):
		self.eps = eps 
		self.momentum = momentum
		self.training = True
		# parameters trained with backprop
		self.gamma = torch.ones(dim)
		self.beta = torch.zeros(dim)
		# buffers
		self.running_mean = torch.zeros(dim)
		self.running_std = torch.ones(dim)

	def __call__(self, x):
		# compute forward pass
		if self.training:
			xmean = x.mean(0, keepdim = True)
			xvar = x.var(0, keepdim=True)
		else:
			xmean = self.running_mean
			xstd = self.running_var

		xhat = (x-mean) / torch.sqrt(xvar+self.eps)
		self.out = self.gamma * xhat + self.beta

		# update buffers
		if self.training:
			with torch.no_grad():
				self.running_mean = (1-self.momentum)* self.running_mean + self.momentum * xmean
				self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
		return self.out 

	def parameters(self):
		return [self.gamma, self.beta] 

class Tanh:

	def __call__(self, x):
		self.out = torch.tanh(x)
		return self.out

	def parameters(self):
		return []



### Simple organized code
blocksize = 3 #this is how much context to recall

### training (train params), dev/validation (train hyperparams) and test splits (80%, 10%, 10%)
def build_dataset(words):
	
	X, Y = [], []
	for w in words:

		# print(w)
		context = [0]*blocksize
		for ch in w+'.':
			ix = stoi[ch]
			X.append(context)
			Y.append(ix)

			#print(''.join(itos[i] for i in context), '--->', itos[ix])
			context = context[1:]+[ix] #remove oldest context, add new context

	X = torch.tensor(X)
	Y = torch.tensor(Y)

	print(X.shape, Y.shape)
	return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr , Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


n_embed = 10
n_hidden = 200
vocab_size = len(chars)+1

C  = torch.randn((vocab_size, n_embed),		      generator=g)
layers = [
	Linear(n_embed*blocksize, n_hidden), BatchNorm1d(n_hidden), Tanh(),
	Linear(			n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
	Linear(			n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
	Linear(			n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
	Linear(			n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
	Linear(			n_hidden, vocab_size),
	
]

with torch.no_grad():
	# last layer less confident
	layers[-1].weight *= 0.1
	# apply gain to all other linear layers
	for layer in layers[:-1]:
		if isinstance(layer, Linear):
			layer.weight *= 5/3 # gain is necessary to prevent tanh squishing the distribution


parameters = [C] + [p for layer in layers for p in layer.parameters()]

print(sum(p.nelement() for p in parameters))

for p in parameters:
	p.requires_grad = True

max_steps = 500000
batch_size = 32
lossi = []
ud = []
for i in range(max_steps):

	#b uild minibatch
	ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)
	Xb, Yb = Xtr[ix], Ytr[ix]

	#forward pass
	emb = C[Xb]
	x = emb.view(emb.shape[0], -1) #concatenate the vectors
	for layer in layers:
		x = layer(x)
	loss = F.cross_entropy(x, Yb)

	# backwards pass
	for layer in layers:
		layer.out.retain_grad()

	for p in parameters:
		p.grad = None
	loss.backward()

	# update
	lr = 0.05 if i < max_steps / 2 else 0.01
	for p in parameters:
		p.data += -lr * p.grad 

	# track stats
	if i % (max_steps/10) == 0:
		print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
	lossi.append(loss.log10().item())
	with torch.no_grad():
		ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])

	if i >= 10000:
		break

# visualize histograms of layer activation
plt.figure(figsize=(20, 4))
legends = []

for i, layer in enumerate(layers[:-1]): #exclude output layer
	if isinstance(layer, Tanh):
		t = layer.out 
		print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, 
				layer.__class__.__name__, 
				t.mean(), t.std(), 
				(t.abs()>0.97).float().mean()*100))
		hy, hx = torch.histogram(t, density=True)
		plt.plot(hx[:-1].detach(), hy.detach())
		legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')
# plt.show()


# histogram of gradients
plt.figure(figsize=(20, 4))
legends = []
for i, layer in enumerate(layers[:-1]): #exclude output layer
	if isinstance(layer, Tanh):
		t = layer.out.grad
		print('layer %d (%10s): mean %+.2f, std %e' % (i, 
				layer.__class__.__name__, 
				t.mean(), t.std()))
		hy, hx = torch.histogram(t, density=True)
		plt.plot(hx[:-1].detach(), hy.detach())
		legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('gradient distribution')


# parameter values visualization
# scale of gradient compared to actual values
plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters): #exclude output layer
	t = p.grad 
	if p.ndim == 2:
		print('weight %10s | mean %+f | std %e | grad:data ration %e' % (tuple(p.shape), 
				t.mean(), t.std(), t.std() / p.std() ))
		hy, hx = torch.histogram(t, density=True)
		plt.plot(hx[:-1].detach(), hy.detach())
		legends.append(f'{i}  {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution')


# at each update, keep track of log of update size compated to data size
# these should be low or else we are over updating
# last layer can be large since that layer was ariticially compressed
plt.figure(figsize=(20,4))
legends = []
for i,p in enumerate(parameters):
	if p.ndim == 2:
		plt.plot([ud[j][i] for j in range(len(ud))])
		legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # ratios should be 1e-3
plt.legend(legends)
plt.show()

# calibrate the batch norm at end of training
# with torch.no_grad():
# 	# pass training set through
# 	emb = C[Xtr]
# 	embcat = emb.view(emb.shape[0], -1)
# 	hpreact = embcat @ W1 + b1
# 	# measure mean/sd over entire training set
# 	bnmean = hpreact.mean(0, keepdim=True)
# 	bnstd = hpreact.std(0, keepdim=True)

# use running approximation instead

# #loss for entire data set
@torch.no_grad() # disable gradient tracking in loss calculation
def split_loss(split):
	x, y = {
		'train': (Xtr, Ytr),
		'val': (Xdev, Ydev),
		'test': (Xte, Yte)
	}[split]
	emb = C[x]
	x = emb.view(emb.shape[0], -1) #concatenate the vectors
	for layer in layers:
		x = layer(x)
	loss = F.cross_entropy(x, y)
	print(split, loss.item())

split_loss('train')
split_loss('val')

## sample names from the model
for i in range(10):
	out = []
	context = [0]*blocksize

	while True:
		emb = C[torch.tensor([context])]
		x = emb.view(emb.shape[0], -1) #concatenate the vectors
		for layer in layers:
			x = layer(x)
		probs = F.softmax(x, dim=1)
		ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()

		context = context[1:]+[ix]
		out.append(itos[ix])
		if ix == 0:
			break


	print(''.join(out))



# for i in range(max_steps):

# 	#b uild minibatch
# 	ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)
# 	Xb, Yb = Xtr[ix], Ytr[ix]

# 	#forward pass
# 	emb = C[Xb]
# 	embcat = emb.view(emb.shape[0], -1) #concatenate the vectors
# 	hpreact = embcat @ W1 #hidden layer pre activation
# 	# want the hidden states to be roughly gaussian
# 	# so normalize them so that it's true - this is a differentiable operation
# 	# subtract mean and divide by sd

# 	# Batch Norm layer
# 	# ----------------------------------------------------------------------------------
# 	bnmeani = hpreact.mean(0, keepdim=True)
# 	bnstdi = hpreact.std(0, keepdim=True)
# 	hpreact = bn_gains*( hpreact - bnmeani) / bnstdi + bn_bias
# 	with torch.no_grad():
# 		bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
# 		bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi  
# 	# ----------------------------------------------------------------------------------
# 	# Non-linear layer
# 	h = torch.tanh(hpreact)
# 	logits = h @ W2 + b2 
# 	loss = F.cross_entropy(logits, Yb)

# 	# backwards pass
# 	for p in parameters:
# 		p.grad = None
# 	loss.backward()

# 	# update
# 	lr = 0.1 if i < max_steps / 2 else 0.01
# 	for p in parameters:
# 		p.data += -lr * p.grad 

# 	# track stats
# 	if i % (max_steps/10) == 0:
# 		print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
# 	lossi.append(loss.log10().item())
