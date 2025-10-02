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


g = torch.Generator().manual_seed(2147483647)

dim_embed = 16
n_hidden = 300
vocab_size = len(chars)+1

# C = torch.randn((vocab_size,dim_embed), generator = g)

# W1 = torch.randn(((blocksize+1)*dim_embed, n_hidden), 	generator=g) * 0.2	# prevent tanh having too extreme values in preactivation
# b1 = torch.randn(n_hidden, 								generator=g) * 0.1 # ditto
# W2 = torch.randn(n_hidden, 								generator=g) * 0.01 # less extreme values in out, better initialization
# b2 = torch.randn(vocab_size, 							generator=g) * 0 	# better initialization loss


C, W1, b1, W2, b2 = torch.load('params_en_min_64.pt')

parameters = [C, W1, b1, W2, b2]
print(sum(p.nelement() for p in parameters)) #total num params

for p in parameters:
	p.requires_grad = True

lri = []
lossi = []
stepi = []

max_steps = 500000

for i in range(max_steps):

	# learning rate
	lr = 0.01 if i < max_steps/2 else 0.001

	#construct minibatch
	mini_batch_size = 64
	ix = torch.randint(0, Xtr.shape[0], (mini_batch_size,), generator=g) # length 32

	#reshape with duplicate rows
	# Xtr[ix] is minibatch x3
	# Xtr[ix].unsqueeze(1) is minibatch x 1 x 3
	# Xtr[ix].unsqueeze(1).repeat(1, 27, 1).view(mini batch,27,3) is minibatch x 27 x blocksize
	A = Xtr[ix].unsqueeze(1).repeat(1,vocab_size,1).view(mini_batch_size,vocab_size,blocksize)
	#add all possible outputs to minimize energy over outputs
	# 32 x 27 x blocksize+1
	B = torch.cat((A, torch.arange(vocab_size).view(vocab_size,1).repeat(mini_batch_size,1,1)), dim=2)
	#forward pass
	# C[B] is 32 x 27 x blocksize+1 x embedding dimension
	# embed.view(-1, *) is 32 x 27 x (blocksize+1 * emb dim)
	embed = C[B]
	# embed.view * W1 + b1 is 32 x 27 x 100
	h = torch.tanh(embed.view(mini_batch_size,vocab_size, (blocksize+1)*dim_embed) @ W1 +b1)
	# energy is 32 x 27
	energy = h @ W2 + b2[A].sum(2)
	# logits = F.softmax(energy, dim = 1)
	### call torch cross entropy loss
	loss = F.cross_entropy(energy, Ytr[ix]) + lr * (W1**2).mean() 

	#backward pass
	for p in parameters:
		p.grad = None
	loss.backward()

	#update
	for p in parameters:
		p.data += -lr * p.grad

	# track stats
	check_points = max_steps / 20
	if i % check_points == 0:
		print(f'{i:7d} / {max_steps:7d}: {loss.item():4f}')

	stepi.append(i)
	lossi.append(loss.log10().item())


plt.plot(stepi, lossi)
plt.show()

# #loss for entire data set
@torch.no_grad() # disable gradient tracking in loss calculation
def split_loss(split):
	x, y = {
		'train': (Xtr, Ytr),
		'val': (Xdev, Ydev),
		'test': (Xte, Yte)
	}[split]
	A = x.unsqueeze(1).repeat(1,vocab_size,1).view(x.shape[0],vocab_size,blocksize)
	B = torch.cat((A, torch.arange(vocab_size).view(vocab_size,1).repeat(x.shape[0],1,1)), dim=2)
	embed = C[B]
	h = torch.tanh(embed.view(x.shape[0], 27, (blocksize+1)*dim_embed) @ W1 +b1)
	energy = h @ W2 + b2[A].sum(2)
	# logits = F.softmax(energy, dim = 1)
	loss = F.cross_entropy(energy, y)
	print(split, loss.item())

split_loss('val')

### sample names from the model
for i in range(10):
	out = []
	context = [0]*blocksize

	while True:
		A = torch.tensor(context).repeat(vocab_size,1)
		B = torch.cat((A, torch.arange(vocab_size).view(vocab_size,1)), dim=1)
		embed = C[B]
		h = torch.tanh(embed.view(vocab_size, (blocksize+1)*dim_embed) @ W1 +b1)
		energy = h @ W2 + b2[A].sum(1)
		probs = F.softmax(energy, dim = 0)
		ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
		context = context[1:]+[ix]
		out.append(itos[ix])
		if ix == 0:
			break

	print(''.join(out))

torch.save(parameters, 'params_en_min_64.pt')