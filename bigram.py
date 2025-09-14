import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparam
batch_size = 32 #batch size is how many independent batches of data the transformer will use for training at one time
block_size = 8  #block-size/context load is the maximum size to train on at once
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#simultaneously predict each element as a function of the previous elements (how much the transformer remembers when making the next prediction)
# in abcd, predict b from context a; predict c from context a,b; predict d from context a,b,c; repeat this unti you get to blocksize

# -----------

#set a seed for replicatability
torch.manual_seed(31415)


#read in the tiny shakespeare data set
with open('input.txt', 'r', encoding='utf-8') as f:
	text = f.read()

#find the 'language' 
chars = sorted(list(set(text)))
vocab_size = len(chars)


#upgrade option, use tiktoken or sentencepiece to use more real-world encodings

#encoder/decoder definition
c_to_i = { ch:i for i, ch, in enumerate(chars)}
i_to_c = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [c_to_i[c] for c in s] 
decode = lambda l: ''.join([i_to_c[i] for i in l])



# create a torch tensor of the dataset
data = torch.tensor(encode(text), dtype=torch.long)
#split into training/validation data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
	# generates a small batch of data with inputs x and output y
	# batch and blocksize are globally defined
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x= torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x,y


# do not call the gradient on anything that happens in here, i.e. no backprop will occur
@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

#Bigram language model
class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table, i.e. the model is just a multiplying a one hot representatin of 
		# each character by the a square matrix, and finding the row corresopnding to the character that is 1
		self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

	def forward(self, idx, targets=None):

		#idx and targets are both (B,T) tensor of integers
		#picks out the row corresponding to the idx
		#B = Batch 4; T = Time 8; C = Channels vocab_size
		logits = self.token_embedding_table(idx) #(B,T,C)

		if targets == None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C) #2d array

			targets = targets.view(B*T)

			#negative log likliehood loss
			#quality of logits compared to the targets
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
		#idx is (B,T) array of indices
		for _ in range(max_new_tokens):
			# get the predictions
			logits, loss = self(idx)  #this is like calling m(idx)

			#only focus on the last time step
			logits = logits[:,-1,:] #just (B, C) dim now

			#apply softmax
			probs = F.softmax(logits, dim=-1) #(B, C)

			#sample from the prob dist
			idx_next = torch.multinomial(probs, num_samples = 1) #(B, 1)

			# append the sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)

		return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)


#start with a new line entry
#idx = torch.zeros((1,1), dtype = torch.long)
#generate 100 new entries after the new line entry idx
# index the 0th row, because we only have 1 batch
# convert from torch.tensor to python list so that decode can run
#print(decode(m.generate(idx, max_new_tokens = 100)[0].tolist()))


# create a PyTorch optimizer
# can get away with higher learning rate for small networks
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)



for iter in range(max_iters):

	# periodically evaluate the loss on the training/validation sets
	if iter % eval_interval == 0:
		losses = estimate_loss()
		print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

	# sample a batch of data
	xb, yb = get_batch('train')

	# evaluation the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
