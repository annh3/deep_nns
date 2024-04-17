import torch
import torch.nn as nn
import math
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F


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


class FeedForwardNetwork(nn.Module):
	def __init__(self,d_model,d_ff,dropout=0.1):
		super(FeedForwardNetwork, self).__init__()
		self.d_model = d_model
		self.d_ff = d_ff
		self.w_1 = nn.Linear(d_model,d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self,x):
		return self.w_2(self.dropout(F.ReLU(self.w_1(x))))

class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
	def __init__(self,
				emb_size: int,
				dropout: float,
				maxlen: int = 5000
		):
		super(PositionalEncoding, self).__init__()
		self.embedding_matrix = torch.zeros((maxlen, emb_size))
		pos = torch.arange(0,maxlen).reshape(maxlen,1) # for broadcasting
		denominator = torch.exp(-torch.arange(0,emb_size,2)*math.log(10000)/emb_size)
		self.embedding_matrix[:, 0::2] = torch.sin(pos * denomiator)
		self.embedding_matrix[:, 1::2] = torch.cos(pos * denominator)
		self.pos_embedding = pos_embedding.unsqueeze(0)

		self.dropout = nn.Dropout(dropout)
		self.register_buffer('pos_embedding', pos_embedding)

	def forward(self, token_embedding: Tensor):
		return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])

class Generator(nn.Module):
	"Linear + softmax generation step."
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)

class LayerNorm(nn.Module):
	def __init__(self, size, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(size))
		self.b_2 = nn.Parameter(torch.zeros(size))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x-mean) / (std + self.eps) + self.b_2 

class SublayerConnection(nn.Module):
	"""
	Residual connection followed by layer norm, where you can 
	pass in the sublayer function, eg mha or ff
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		return self.norm(x + self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size,dropout), 2)
		self.size = size # d_model

	def forward(self, x, mask):
		x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
		return self.sublayer[1](x, feed_foward)

class Encoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size) # d_model # is this necessary?

	def forward(self, x, mask):
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)


class DecoderLayer(nn.Module):
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size,dropout), 3)
		self.size = size # d_model

	def forward(self, x, encoder_outputs, source_mask, target_mask):
		m = encoder_outputs
		x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,target_mask))
		x = self.sublyaer[1](x, lambda x: self.self_attn(x,m,m,source_mask))
		return self.sublayer[2](x, feed_foward)

class Decoder(nn.Module):
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size) # d_model # is this necessary?

	def forward(self, x, encoder_outputs, source_mask, target_mask):
		for layer in self.layers:
			x = layer(x, encoder_outputs, source_mask, target_mask)
		return self.norm(x)


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(V, batch_size, nbatches, d_model=10):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, d_model)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)


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
