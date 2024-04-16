import torch
import torch.nn as nn
import math
import numpy as np


def attention(query, key, value, mask=None, dropout=None):
	# query: (N, D_k)
	# key: (N, L, D_k)
	# value: (N, L, D_v)
	d_k = query.size()[1]
	key_transpose = key.transpose(2,1)
	query = query.unsqueeze(1)
	print("key transpose size: ", key_transpose.size())
	print("query size: ", query.size())
	attention_weights = torch.matmul(query, key_transpose)
	print("atten_weights size: ", attention_weights.size())
	attention_weights = torch.softmax(attention_weights / math.sqrt(d_k), dim=1)
	# (N,L)
	print("atten_weights size: ", attention_weights.size())
	context_vector = torch.matmul(attention_weights, value)
	# (N, D_v)
	print("result size: ", context_vector.size())
	context_vector = context_vector.squeeze(1)
	return context_vector

def attention2(query, key, value, mask=None, dropout=None):
	# query: (N, L, D_k)
	# key: (N, L, D_k)
	# value: (N, L, D_v)
	d_k = query.size()[1]
	attention_weights = torch.einsum('nld,nld->nll', query, key)
	print("atten_weights size: ", attention_weights.size())
	attention_weights = torch.softmax(attention_weights / math.sqrt(d_k), dim=1)
	# (N,L)
	print("atten_weights size: ", attention_weights.size())
	context_vector = torch.einsum('nl,nld->nd', attention_weights, value)
	# (N, D_v)
	print("result size: ", context_vector.size())
	return context_vector


def main():
	N = 100
	L = 10
	d_k = 7
	d_v = 5
	query = torch.randn(N,d_k)
	key = torch.randn(N,L,d_k)
	value = torch.randn(N,L,d_v)
	context = attention2(query,key,value)
	context = context.detach().numpy()
	print(context.shape)
	assert context.shape == (N,d_v)
	context2 = attention2(query,key,value)
	context2 = context2.detach().numpy()
	print(context2.shape)
	assert context2.shape == (N,d_v)
	assert np.allclose(context,context2)


if __name__ == "__main__":
	main()
