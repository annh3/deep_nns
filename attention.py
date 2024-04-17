import torch
import torch.nn as nn
import math
import numpy as np
import copy


def clones(module, N):
	"Produces N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def future_mask(size): # usually size = L, the sequence length
	"""
	[[1 0 0]
	[1 1 0]
	[1 1 1]]
	"""
	attn_shape = (1,size,size)
	mask_tensor = torch.ones(attn_shape)
	mask_tensor = torch.tril(mask_tensor)
	return mask_tensor


def attention(query, key, value, mask=None, dropout=None):
	# query: (N, L, D_k)
	# key: (N, L, D_k)
	# value: (N, L, D_v)
	d_k = query.size()[2]
	key_transpose = key.transpose(2,1)
	attention_weights = torch.matmul(query, key_transpose)

	if mask is not None:
		"""
		[[a -inf -inf]
		[b c -inf]
		[d e f]]
		"""
		attention_weights.masked_fill(mask == 0, 1e-9)

	attention_weights = torch.softmax(attention_weights / math.sqrt(d_k), dim=1)
	# (N,L)
	context_vector = torch.matmul(attention_weights, value)
	# (N, D_v)
	return context_vector

class MultiHeadedAttention(nn.Module):
	def __init__(self,h,d_model,dropout=0.1):
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		self.h = h
		self.d_k = d_model // h 
		self.d_model = d_model
		# Define projection matrices
		self.Q_linears = clones(nn.Linear(self.d_model, self.d_k),h)
		self.K_linears = clones(nn.Linear(self.d_model, self.d_k),h)
		self.V_linears = clones(nn.Linear(self.d_model, self.d_k),h)
		self.final_W = nn.Linear(self.d_model, self.d_model)
		self.dropout = nn.Dropout(dropout)
		self.attention = None

	def forward(self, query, key, value, mask=None):
		# same mask is appled to akk h heads
		N = query.size(0)
		Q_projected = [f(query) for f in self.Q_linears]
		K_projected = [f(query) for f in self.K_linears]
		V_projected = [f(query) for f in self.V_linears]
		attn_contexts = [attention(q,k,v,mask) for q,k,v in zip(Q_projected,K_projected,V_projected)]
		# each of the h attn_contexts should be of dim N x L x d_k
		# we should concatenate them along the last dimension
		attn_context = torch.cat(attn_contexts,dim=-1)
		out = self.final_W(attn_context)
		out = self.dropout(out)
		return out



def main():
	N = 100
	L = 10
	d_model = 64
	d_v = d_model
	query = torch.randn(N,L,d_model)
	key = torch.randn(N,L,d_model)
	value = torch.randn(N,L,d_model)
	context = attention(query,key,value)
	context = context.detach().numpy()
	print(context.shape)
	assert context.shape == (N,L,d_v)

	h = 8

	# testing mha without mask first
	mha = MultiHeadedAttention(h,d_model)
	context = mha(query,key,value)
	context = context.detach().numpy()
	print(context.shape)
	assert context.shape == (N,L,d_v)

	mask = future_mask(L)
	context = mha(query,key,value,mask)
	context = context.detach().numpy()
	print(context.shape)
	assert context.shape == (N,L,d_v)



if __name__ == "__main__":
	main()
