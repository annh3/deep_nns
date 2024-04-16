import torch
import torch.nn as nn
import math
import numpy as np


def attention(query, key, value, mask=None, dropout=None):
	# query: (N, L, D_k)
	# key: (N, L, D_k)
	# value: (N, L, D_v)
	d_k = query.size()[2]
	key_transpose = key.transpose(2,1)
	print("key transpose size: ", key_transpose.size())
	print("query size: ", query.size())
	attention_weights = torch.matmul(query, key_transpose)
	print("atten_weights size: ", attention_weights.size())

	if mask:
		mask_tensor = torch.full((query.size()[1],query.size()[1]),1e-9)
		mask_tensor = torch.triu(mask_tensor)
		mask_tensor = torch.fill_diagonal_(0)
		attention_weights = mask_tensor + attention_weights

	attention_weights = torch.softmax(attention_weights / math.sqrt(d_k), dim=1)
	# (N,L)
	print("atten_weights size: ", attention_weights.size())
	context_vector = torch.matmul(attention_weights, value)
	# (N, D_v)
	print("result size: ", context_vector.size())
	return context_vector



def main():
	N = 100
	L = 10
	d_k = 7
	d_v = 5
	query = torch.randn(N,L,d_k)
	key = torch.randn(N,L,d_k)
	value = torch.randn(N,L,d_v)
	context = attention(query,key,value)
	context = context.detach().numpy()
	print(context.shape)
	assert context.shape == (N,L,d_v)


if __name__ == "__main__":
	main()
