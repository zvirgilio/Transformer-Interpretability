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


### training (train params), dev/validation (train hyperparams) and test splits (80%, 10%, 10%)
def build_dataset(words):
	blocksize = 3 #this is how much context to recall
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

dim_embed = 12

# C = torch.randn((27,dim_embed), generator = g)

# W1 = torch.randn((3*dim_embed, 200), generator=g) #increase hidden layer size to 300
# b1 = torch.randn(200, generator=g)
# W2 = torch.randn((64, 27), generator=g)
# b2 = torch.randn(27, generator=g)

# ### 3rd hidden layer
# W3 = torch.randn((200, 64), generator=g)
# b3 = torch.randn(64, generator=g)

C, W1, b1, W2, b2, W3, b3 = torch.load('params.pt')

parameters = [C, W1, b1, W2, b2, W3, b3]
print(sum(p.nelement() for p in parameters)) #total num params

for p in parameters:
	p.requires_grad = True

# # determine learning rate
# lre = torch.linspace(-3, 0, 1000) #exponents of learning rate
# lrs = 10**lre # learning rate steps

# learning rate decay
# lrs = torch.linspace(2, 0.001, 1500000)
# lrs = torch.linspace(0.1, 0.01, 1500000)

lri = []
lossi = []
stepi = []
for i in range(1):

	#construct minibatch
	ix = torch.randint(0, Xtr.shape[0], (128,), generator=g)

	#forward pass
	embed = C[Xtr[ix]] #32 x 3 x 12
	h1 = torch.tanh(embed.view(-1, 3*dim_embed) @ W1 +b1) #reshape to 32 x 36, W1 is 36 x 200
	h2 = torch.tanh(h1 @ W3 + b3)

	logits = h2 @ W2 + b2
	### call torch cross entropy loss
	loss = F.cross_entropy(logits, Ytr[ix]) + 0.001 * (W1**2).mean() + 0.001 * (W2**2).mean() + 0.001 * (W3**2).mean()
	
	#backward pass
	for p in parameters:
		p.grad = None
	loss.backward()

	#update
	# lr = lrs[i]
	# lr = 0.1 if i < 300000 else 0.01
	lr = 0.001
	for p in parameters:
		p.data += -lr * p.grad

	# # track the stats
	# lri.append(lre[i])
	# lossi.append(loss.item())
	stepi.append(i)
	lossi.append(loss.log10().item())

# plt.plot(lri, lossi)
# plt.show()
plt.plot(stepi, lossi)
plt.show()

# #loss for entire data set
embed = C[Xdev]
h1 = torch.tanh(embed.view(-1, 3*dim_embed) @ W1 +b1)
h2 = torch.tanh(h1 @ W3 + b3)
logits = h2 @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
print(loss.item())

### sample names from the model
for i in range(10):
	out = []
	context = [0]*3

	while True:
		embed = C[torch.tensor([context])]
		h1 = torch.tanh(embed.view(-1, 3*dim_embed) @ W1 +b1)
		h2 = torch.tanh(h1 @ W3 + b3)
		logits = h2 @ W2 + b2
		probs = F.softmax(logits, dim=1)
		ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()

		context = context[1:]+[ix]
		out.append(itos[ix])
		if ix == 0:
			break

	print(''.join(out))

torch.save(parameters, 'params.pt')

# #visualize 2-dim embeddings
# plt.figure(figsize=(8,8))
# plt.scatter(C[:,0].data, C[:, 1].data, s=200)
# for i in range(C.shape[0]):
# 	plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
# plt.grid('minor')
# plt.show()


### training (train params), dev/validation (train hyperparams) and test splits (80%, 10%, 10%)

### embed characters into a space of dimension m (i.e. m=2, 5 etc.), m can be much lower than vocab/alph size 
### use optimization (backprop) to 'move' these embeddings around so that similar characters are nearby in the space
### train using neg log likelihood

# blocksize = 3 #this is how much context to recall

# X, Y = [], []
# for w in words[:5]:

# 	# print(w)
# 	context = [0]*blocksize
# 	for ch in w+'.':
# 		ix = stoi[ch]
# 		X.append(context)
# 		Y.append(ix)

# 		#print(''.join(itos[i] for i in context), '--->', itos[ix])
# 		context = context[1:]+[ix] #remove oldest context, add new context

# X = torch.tensor(X)
# Y = torch.tensor(Y)

# ### build look up table for embedding in a 2-dim space
# C = torch.randn((27,2), generator = g)

# ### index using a tensor ([size X] x 3 x 2) tensor
# embed = C[X]

# ### hidden layer
# # 100 neurons (chosen arbitrarily)
# # inputs into this layer is 6 = 3*2 (3 pieces of context, each embedded in R^2)
# W1 = torch.randn((6, 100)) 
# b1 = torch.randn(100)

# ### reshape because embedding is stacked as 3x2

# ### concatenate along a given dimension
# # torch.cat(embed[:, 0, :], embed[:, 1, :], embed[:, 2, :], 1)

# # unbind removes a dimension, a more general version of the above code
# # torch.cat(torch.unbind(embed, 1), 1)

# # instead we will use .view, a very efficient operation
# # .storage is just all entries as a list
# # .view changes how the list is interpreted, doesn't change/move any data 
# # stacks the 3x2 as a 6x1
# # embed = embed.view(embed.shape[0],6)
# embed = embed.view(-1, 6) # -1 lets torch infer

# # remember to double check the torch broadcasting
# ### hidden layer
# h = torch.tanh(embed @ W1 +b1)

# W2 = torch.randn((100, 27))
# b2 = torch.randn(27)

# logits = h @ W2 + b2
# counts = logits.exp()
# probs = counts / counts.sum(1, keepdims = True)

# #check prob of actual values
# #print(probs[torch.arange(32), Y])

# loss = -probs[torch.arange(32), Y].log().mean()
# print(loss)


