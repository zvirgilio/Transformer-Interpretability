import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt 
import random
import math 
import time
import numpy as np

#local imports
from TransformerModules import Modelconfig, Block, MLP, CausalSelfAttention

class AttentionOnlyModel(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config 
		self.transformer = nn.ModuleDict(
			dict(
				wce = nn.Embedding(config.vocab_size, config.n_embed),
				wpe = nn.Embedding(config.block_size, config.n_embed),
				h = CausalSelfAttention(config)
				)
			)
		self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)


	def forward(self, idx, targets = None):
		B, T = idx.shape

		assert T <= self.config.block_size, f"Cannot forward sequence of lenght {T}, block size is {self.config.block_size}"

		char_emb = self.transformer.wce(idx)
		pos_emb = self.transformer.wpe(torch.arange(0, T, dtype = torch.long))

		x = char_emb + pos_emb

		x = x + self.transformer.h(x)

		logits = self.lm_head(x)

		if targets == None:
			loss = None 

		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)

			targets = targets.view(B*T)

			loss = F.cross_entropy(logits, targets)

		return logits, loss


	def generate(self, idx, max_new_tokes):

		for _ in range(max_new_tokes):

			idx_cond = idx[:, -self.config.block_size:]

			logits, loss = self(idx_cond)

			logits = logits[:, -1, :]

			probs = F.softmax(logits, dim = -1)

			idx_next = torch.multinomial(probs, num_samples = 1)

			idx = torch.cat((idx, idx_next), dim=1)

		return idx 



#--------------------------------------------------------------------------------------------
# tools to build the basic A, B, C, A, B, C, ... patterned dataset
data = list('ABC'*1000)
c_to_i = {'A':0, 'B':1, 'C':2}
i_to_c = {i:c for c,i in c_to_i.items()}
encode = lambda s: [c_to_i[c] for c in s] 
decode = lambda l: ''.join([i_to_c[i] for i in l])


i_data = [c_to_i[c] for c in data]
i_data = torch.tensor(i_data, dtype=torch.long)




# get a batch of training data
def get_batch(model, batch_size):
	block_size = model.config.block_size
	ix = torch.randint(len(i_data)-block_size, (batch_size,))
	x = torch.stack([i_data[i:i+block_size] for i in ix])
	y = torch.stack([i_data[i+1:i+block_size+1] for i in ix])
	return x,y


@torch.no_grad()
def estimate_loss(model, batch_size, train_data, val_data, eval_iters):
	out = {}
	data = {'train': train_data, 'val': val_data}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(model, batch_size)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

def training_loop(model, optimizer, batch_size, max_iters, eval_interval, train_data, val_data, eval_iters):
	start_time = time.time()

	for iter in range(max_iters):

		if iter % eval_interval == 0:
			losses = estimate_loss(model, batch_size, train_data, val_data, eval_iters)
			print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

		# sample a batch of data
		xb, yb = get_batch(model, batch_size)

		# evaluation the loss
		logits, loss = model(xb, yb)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

	losses = estimate_loss(model, batch_size, train_data, val_data, eval_iters)
	print(f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
	print(f"Took {(time.time() - start_time)//60} minutes to train")	


def main():
	tinyconfig = Modelconfig
	tinyconfig.block_size = 3
	tinyconfig.vocab_size = 3
	tinyconfig.n_layer = 1
	tinyconfig.n_head = 1
	tinyconfig.n_embed = 2

	torch.manual_seed(1234)

	one_layer_model = AttentionOnlyModel(tinyconfig)
	optimizer = torch.optim.AdamW(one_layer_model.parameters(), )

	max_iters=5000
	# eval_interval=max_iters//100
	eval_interval = 500
	eval_iters = 200
	batch_size = 64

	print("beginning training")
	training_loop(one_layer_model, optimizer, batch_size, max_iters, eval_interval, i_data, i_data, eval_iters)

	context = torch.zeros((1,1), dtype = torch.long)
	print(decode(one_layer_model.generate(context, 20)[0].tolist()))

	for param_tensor in one_layer_model.state_dict():
		print(param_tensor, "\t", one_layer_model.state_dict()[param_tensor])




if __name__ == '__main__':
	main()