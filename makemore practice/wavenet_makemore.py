import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
words = open('names.txt', 'r').read().splitlines()

#build the vocabulary of characters and mapping to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

vocab_size = len(itos)

### Simple organized code
blocksize = 8 #this is how much context to recall

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


## classes keep a self.out to be able to visualize and keep track of activation
#---------------------------------------------------------------------------------------------------------------------------
class Linear:

	def __init__(self, fan_in, fan_out, bias=True):
		self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
		self.bias = torch.zeros(fan_out) if bias else None

	def __call__(self, x):
		self.out = x @ self.weight
		if self.bias is not None:
			self.out += self.bias 
		return self.out 

	def parameters(self):
		return [self.weight] + ([] if self.bias is None else [self.bias])

#---------------------------------------------------------------------------------------------------------------------------
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
		self.running_var = torch.ones(dim)

	def __call__(self, x):
		# compute forward pass

		# mean and var along dimension 0 is a result of assuming a 2d input
		# can input a tuple of dimensions to take mean over
		# we want the first two dimensions to be the 'batch' dims
		if x.ndim == 2:
			dim = 0
		elif x.ndim == 3:
			dim = (0,1)

		if self.training:
			xmean = x.mean(dim, keepdim = True) #batch mean
			xvar = x.var(dim, keepdim=True)	  #batch var
		else:
			xmean = self.running_mean
			xvar = self.running_var

		xhat = (x-xmean) / torch.sqrt(xvar+self.eps)
		self.out = self.gamma * xhat + self.beta

		# update buffers
		if self.training:
			with torch.no_grad():
				self.running_mean = (1-self.momentum)* self.running_mean + self.momentum * xmean
				self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
		return self.out 

	def parameters(self):
		return [self.gamma, self.beta] 

#---------------------------------------------------------------------------------------------------------------------------
class Tanh:

	def __call__(self, x):
		self.out = torch.tanh(x)
		return self.out

	def parameters(self):
		return []

#---------------------------------------------------------------------------------------------------------------------------
class Embedding:

	def __init__(self, num_embeddings, embedding_dim):
		self.weight = torch.randn((num_embeddings, embedding_dim))

	def __call__(self, IX):
		self.out = self.weight[IX]
		return self.out

	def parameters(self):
		return [self.weight]

#---------------------------------------------------------------------------------------------------------------------------
#flattens n conesecutive elements and puts them in the last dimension
#i.e. for n=2, could take a (4,8,10) object (i.e. batch size 4, 8 letters in each block, 10 dimensions on each vector) and makes it a
# (4,4,20) by taking the first two 10-dim vectors for each letter in a block, and concatenating them, and likewise 3rd and 4th, 5th and 6th ...
class FlattenConsecutive:

	def __init__(self, n):
		self.n = n
	# def __call__(self, x):
	# 	self.out = x.view(x.shape[0], -1) # concatenates all the vectors corresponding to a block into a single long vector
	# 	return self.out
	def __call__(self, x):
		B, T, C = x.shape
		x = x.view(B, T//self.n, C*self.n)
		if x.shape[1] == 1: 	# if this dim vanishes, i.e. is spurious
			x = x.squeeze(1)	# squeeze it out

		self.out = x 
		return self.out


	def parameters(self):
		return []

#---------------------------------------------------------------------------------------------------------------------------
class Sequential:

	def __init__(self, layers):
		self.layers = layers

	def __call__(self, x):
		for layer in self.layers:
			x = layer(x)

		self.out = x
		return self.out 

	def parameters(self):
		# get all parameters and put them in a single list
		return [p for layer in self.layers for p in layer.parameters()]

#---------------------------------------------------------------------------------------------------------------------------
torch.manual_seed(42) #for reporoducibility without a generator

n_embed = 24
n_hidden = 128 #yields roughly same number of parameters as flat mlp, can see if there is better performance because 
			  # of a better architecture (heirarchical mlp)


model = Sequential([
	Embedding(vocab_size, n_embed), 
	FlattenConsecutive(2),
	Linear(n_embed * 2, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(), 
	FlattenConsecutive(2),
	Linear(n_hidden * 2, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(), 
	FlattenConsecutive(2),
	Linear(n_hidden * 2, n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(), 
	Linear(n_hidden, vocab_size),
])

# parameter initialization
with torch.no_grad():
	model.layers[-1].weight *= 0.1 #make last layer less confident at init

parameters = model.parameters()
print(sum(p.nelement() for p in parameters)) # total parameters
for p in parameters:
	p.requires_grad = True

# training stage
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

	# construct minibatch
	ix = torch.randint(0, Xtr.shape[0], (batch_size, ))
	Xb, Yb = Xtr[ix], Ytr[ix]

	#forward pass
	logits = model(Xb)

	loss = F.cross_entropy(logits, Yb)

	# backwards pass
	for p in parameters:
		p.grad = None
	loss.backward()

	# update with simple sgd
	lr = 0.01 if i < 150000 else 0.01
	for p in parameters:
		p.data += -lr * p.grad 

	# track stats
	if i % (max_steps / 20) == 0:
		print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
	lossi.append(loss.log10().item())


# verify all shapes are correct
for layer in model.layers:
	print(layer.__class__.__name__, ':', tuple(layer.out.shape))

# print loss (average of every 1000 batches)
lossi = torch.tensor(lossi)
lossi = lossi.view(-1, 1000).mean(dim=1)
plt.plot(lossi)
plt.show()

# put layers in eval mode
for layer in model.layers:
	layer.training = False

#loss for entire data set
@torch.no_grad() # disable gradient tracking in loss calculation
def split_loss(split):
	x, y = {
		'train': (Xtr, Ytr),
		'val': (Xdev, Ydev),
		'test': (Xte, Yte)
	}[split]
	logits = model(x)
	loss = F.cross_entropy(logits, y)
	print(split, loss.item())

split_loss('train')
split_loss('val')


# sample from the model
for _ in range(20):

	out=[]
	context = [0]*blocksize
	while True:
		# forward pass
		logits = model(torch.tensor([context]))
		probs = F.softmax(logits, dim=1)

		# sample
		ix = torch.multinomial(probs, num_samples=1).item()

		context = context[1:]+[ix]
		out.append(ix)

		if ix == 0:
			break

	print(''.join(itos[i] for i in out))








