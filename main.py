from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F 

#----------------------------------------------------------------------------------------------
@dataclass 
class GPTconfig:
	block_size: int = 256
	vocab_size: int = 65
	n_layer: int = 6
	n_head: int = 4
	n_embed: int = 256

#----------------------------------------------------------------------------------------------	
class GPT(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.config = config 

		#can store the state dict as a dictionary of modules, able to index into components
		#wte: weight token embed
		#wpe: weight position embed
		#h: hidden, this is a list where each layer can be indexed with a number
		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embed),
			wpe = nn.Embedding(config.block_size, config.n_embed),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			ln_f = nn.LayerNorm(config.n_embed),
			))

		self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

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
		B, T, C = x.size() #batch size, sequence length, embedding dimension 

		# calculate query, key and values for all heads in a batch and move head forward to be the batch
		# nh: "number of heads", hs: "head size" and C: "number of channels" = nh * ns


