import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import random
import time
import math
from dataclasses import dataclass 

eval_iters=200

#----------------------------------------------------------------------------------------------
@dataclass
class Namerconfig:
	block_size: int = 8
	vocab_size: int = 27
	n_layer: int = 4
	n_head: int = 4
	n_embed: int = 16


#----------------------------------------------------------------------------------------------
class Namer(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config 


		self.transformer = nn.ModuleDict(
			dict(
				wce = nn.Embedding(config.vocab_size, config.n_embed),					# weights of char embedding
				wpe = nn.Embedding(config.block_size, config.n_embed),					# weights of position embedding
				h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),		# hidden attention layers
				ln_f = nn.LayerNorm(config.n_embed),
				)
			)

		self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)


	def forward(self, idx, targets = None):
		B, T = idx.shape
		assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

		char_emb =  self.transformer.wce(idx)

		pos_emb = self.transformer.wpe(torch.arange(0, T, dtype=torch.long))

		x = char_emb + pos_emb

		for block in self.transformer.h:
			x = block(x)

		x = self.transformer.ln_f(x)

		logits = self.lm_head(x)

		if targets == None:
			loss = None

		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)

			targets = targets.view(B*T)

			loss = F.cross_entropy(logits, targets)

		return logits, loss 

	def generate(self):
		
		out=[]
		context = torch.zeros((1,8), dtype=torch.long)
		while True:
			# forward pass
			logits, loss = self(context[:,-self.config.block_size:])
				
			logits = logits[:, -1, :]

			probs = F.softmax(logits, dim=-1)
						# sample
			ix = torch.multinomial(probs, num_samples=1).item()

			context = torch.cat((context, torch.tensor([ix]).view(1,1)), dim=1)
			out.append(ix)

			if ix == 0:
				break

		name = ''.join(itos[i] for i in out)
		return name

#----------------------------------------------------------------------------------------------
class Block(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embed)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embed)
		self.mlp = MLP(config)


	def forward(self, x):
		''' x + attention + mlp( x + attention )
			keeps a residual stream and forks off the learned block
			'''
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))

		return x

#----------------------------------------------------------------------------------------------
class MLP(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.c_fc 	= nn.Linear(config.n_embed, 4 * config.n_embed)
		self.gelu 	= nn.GELU()	# gelu fixes 0 gradient component of ReLU, no approximate since doesn't need speed up nowadays
		self.c_proj	= nn.Linear(config.n_embed * 4, config.n_embed)


	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x

#----------------------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

	def __init__(self, config):
		super().__init__()

		assert config.n_embed % config.n_head == 0

		# key, query, value projections
		self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)

		# output projection
		self.c_proj = nn.Linear(config.n_embed, config.n_embed)

		# regularization
		self.n_head = config.n_head
		self.n_embed = config.n_embed

		# maks more than buffer
		self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
										  .view(1, 1, config.block_size, config.block_size))


	def forward(self, x):
		B, T, C = x.size() #batch size, sequence length (block_size), embedding dimension (n_embed)

		# calculate query, key and values for all heads in a batch and move head forward to be the batch
		# nh: "number of heads", hs: "head size" and C: "number of channels" = nh * ns
		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.n_embed, dim=2)

		k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)		# B x nh x T x hs
		q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)		# B x nh x T x hs
		v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)		# B x nh x T x hs

		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))		# (B x nh x T x hs) @ (B x nh x hs x T) = B x nh x T x T
		att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
		att = F.softmax(att, dim = -1)
		y = att @ v 														# (B x nh x T x T) @ (B x nh x T x hs) = B x nh x T x hs
		y = y.transpose(1, 2).contiguous().view(B, T, C)					# re-assemble all head outputs side by side

		# output projection
		y = self.c_proj(y)
		return y 

#----------------------------------------------------------------------------------------------

#build the vocabulary of characters and mapping to/from integers
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


def build_dataset(words, config):
	data = []
	for w in words:
		data = data + ([0]*config.block_size) + encode(w)

	data = torch.tensor(data, dtype = torch.long)
	return data 




def get_batch(data, block_size, batch_size):
	# generates a small batch of data with inputs x and output y
	ix = torch.randint(len(data) - block_size, (batch_size,))
	x= torch.stack([data[i:i+block_size] for i in ix])
	y = torch.stack([data[i+1:i+block_size+1] for i in ix])
	return x,y


@torch.no_grad()
def estimate_loss(model, batch_size, train_data, val_data):
	out = {}
	data = {'train': train_data, 'val': val_data}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(data[split], model.config.block_size, batch_size)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out


def training_loop(model, optimizer, batch_size, max_iters, eval_interval, train_data, val_data):
	start_time = time.time()

	for iter in range(max_iters):

		if iter % eval_interval == 0:
			losses = estimate_loss(model, batch_size, train_data, val_data)
			print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

		# sample a batch of data
		xb, yb = get_batch(train_data, model.config.block_size, batch_size)

		# evaluation the loss
		logits, loss = model(xb, yb)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

	losses = estimate_loss(model, batch_size, train_data, val_data)
	print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
	print(f"Took {(time.time() - start_time)//60} minutes to train")	



def main():
	torch.manual_seed(1234)
	config = Namerconfig
	model = Namer(config)

	print("initialized model")

	random.seed(42)
	random.shuffle(words)
	n = int(0.9*len(words))

	train_data = build_dataset(words[:n], config)
	val_data = build_dataset(words[n:], config)
	print("built data sets")

	optimizer = torch.optim.AdamW(model.parameters(), )

	max_iters=5000
	# eval_interval=max_iters//100
	eval_interval = 500
	batch_size = 64

	print("beginning training")
	training_loop(model, optimizer, batch_size, max_iters, eval_interval, train_data, val_data)

	for _ in range(10):
		print(model.generate())

if __name__ == '__main__':
	main()


