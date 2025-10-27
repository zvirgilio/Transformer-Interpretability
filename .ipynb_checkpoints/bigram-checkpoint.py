import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparam
batch_size = 64 #batch size is how many independent batches of data the transformer will use for training at one time
block_size = 128  #block-size/context load is the maximum size to train on at once
max_iters = 15000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 256 # 384/6 = 64 means each head has size 64
num_heads = 4
n_layer = 4 
dropout = 0.2
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

#----------------------------------------------------------------------------------------------------------
class Head(nn.Module):
	''' one head of self attention '''

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias=False)
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		# tril is not a parameter of the module, so in pytorch naming conventions, its a buffer
		# assign it to the module with a register buffer
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x) 	#(B, T, C) these are only C while head_size = n_embed = C
		q = self.query(x)	#(B, T, C)

		# compute attention scores (affinities)
		wei = q @ k.transpose(-2, -1) * C**-0.5 	# (B, T, C) @ (B, C, T) --> (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		wei = F.softmax(wei, dim=-1) #(B, T, T)
		wei = self.dropout(wei)
		# perform weighted aggregation of the values
		v = self.value(x)	# (B, T, C)
		out = wei @ v 		# (B, T, T) @ (B, T, C) --> (B, T, C)

		return out 

#----------------------------------------------------------------------------------------------------------	
class MultiHeadAttention(nn.Module):
	''' multiple heads of self-attention in parallel '''

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embed, n_embed) #projection back into the residual pathway
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out)) #apply the projection
		return out

#----------------------------------------------------------------------------------------------------------	
class FeedForward(nn.Module):
	'''a simple linear layer followed by non-linearity'''

	def __init__(self, n_embed):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embed, n_embed * 4), #factor of 4 to channel sizes in feedforward.  Added computation in resid block
			nn.ReLU(),
			nn.Linear(4 * n_embed, n_embed), # projection layer back into residual pathway
			nn.Dropout(dropout)				 # add before residual function
		)

	def forward(self, x):
		return self.net(x)

#----------------------------------------------------------------------------------------------------------
class Block(nn.Module):
	''' Transformer block: communication followed by computation'''

	def __init__(self, n_embed, num_heads):
		# n_embed: embedding dim, num_heads: number of heads in each multihead attention layer
		super().__init__()
		head_size = n_embed // num_heads
		self.sa = MultiHeadAttention(num_heads, head_size)
		self.ffwd = FeedForward(n_embed)
		self.ln1 = nn.LayerNorm(n_embed)	# normalize the layers (rows) before attention and again before feedforward
		self.ln2 = nn.LayerNorm(n_embed)

	def forward(self, x):
		# x = self.sa(x)
		# x = self.ffwd(x)
		x = x + self.sa(self.ln1(x)) # the self attention heads fork off to the side and are added back, these are the resid connections
		x = x + self.ffwd(self.ln2(x))
		return x

#----------------------------------------------------------------------------------------------------------
#Bigram language model
class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		# each token directly reads off the logits for the next token from a lookup table, i.e. the model is just a multiplying a one hot representatin of 
		# each character by the a square matrix, and finding the row corresopnding to the character that is 1
		self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
		self.position_embedding_table = nn.Embedding(block_size, n_embed) #embeds the importance of each position in the block
		
		#self.sa_head = Head(n_embed) # create the self attention head, keep head size as n_embed
		# self.sa_heads = MultiHeadAttention(4, n_embed//4) # 4 - heads each with size 8 (since n_embed = 32) 
		# self.ffwd = FeedForward(n_embed)

		self.blocks = nn.Sequential(*[Block(n_embed, num_heads=num_heads) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embed) # final layer norm
		self.lm_head = nn.Linear(n_embed, vocab_size) # linear model head, this linear layer goes from embeddings to logits

	def forward(self, idx, targets=None):
		B, T = idx.shape


		#idx and targets are both (B,T) tensor of integers
		#picks out the row corresponding to the idx
		#B = Batch 4; T = Time 8; C = Channels vocab_size
		token_embed = self.token_embedding_table(idx) #(B,T,C)
		pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
		x = token_embed+pos_embed
		# x = self.sa_heads(x)     #(B, T, C)
		# x = self.ffwd(x)		 #(B, T, C)
		x = self.blocks(x) 		 #(B, T, C)
		x = self.ln_f(x)		 #(B, T, C)
		# at this point, the network is getting deep and optimization is becoming complex
		#use residual pathways to help optimization
		
		logits = self.lm_head(x) #(B, T, vocab_size)

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
			# crop idx to the last block_size token
			# if idx is more than block_size, pos_embedding table will not be able to assign it a value
			idx_cond = idx[:, -block_size:]

			# get the predictions
			logits, loss = self(idx_cond)  #this is like calling m(idx)

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

start_time = time.time()

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

losses = estimate_loss()
print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print(f"Took {(time.time() - start_time)//60} minutes to train")

# generate from the model
context = torch.zeros((1,1), dtype = torch.long, device=device)
with open("shakespeare_gibberish.txt", "w") as f:
	f.write(decode(m.generate(context, max_new_tokens=1500)[0].tolist()))
# print(decode(m.generate(context, max_new_tokens=1500)[0].tolist()))
torch.save(model.state_dict(), 'bigram_model_params')