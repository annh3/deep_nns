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
    #print("future mask size: ", size)
    attn_shape = (1,size,size)
    mask_tensor = torch.ones(attn_shape)
    mask_tensor = torch.tril(mask_tensor)
    return mask_tensor


def attention(query, key, value, mask=None, dropout=None):
    # query: (N, L, D_k)
    # key: (N, L, D_k)
    # value: (N, L, D_v)
    d_k = query.size(-1)
    key_transpose = key.transpose(-2,-1)
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

class CachedMultiHeadedAttention(nn.Module):
   # https://r4j4n.github.io/blogs/posts/kv/
    def __init__(self,h,d_model,max_seq_len,dropout=0.1):
        super(CachedMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        # create separate keys cache for each mha head
        self.keys_cache = [torch.zeros(1,max_seq_len,self.d_k) for _ in range(h)]
        self.values_cache = [torch.zeros(1,max_seq_len,self.d_k) for _ in range(h)]
        self.Q_matrices = clones(nn.Linear(d_model,self.d_k),h)
        self.K_matrices = clones(nn.Linear(d_model,self.d_k),h)
        self.V_matrices = clones(nn.Linear(d_model,self.d_k),h)
        self.final_linear = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None, pos=0):
        # N x L x D
        # batch_size, L, dimension

        # projecting the new input at position pos
        x_Q_projected = [self.Q_matrices[i](x) for i in range(self.h)]
        x_K_projected = [self.K_matrices[i](x[:,-1,:]) for i in range(self.h)]
        x_V_projected = [self.V_matrices[i](x[:,-1,:]) for i in range(self.h)]
        for i in range(self.h):
            self.keys_cache[i][:,pos,:] = x_K_projected[i]
            self.values_cache[i][:,pos,:] = x_V_projected[i]

        # need to concatenate with the previous keys and values
        # for each mha h, need to fetch from cache and project with the new x
        attn_contexts = []
        for i in range(self.h):
            keys = self.keys_cache[i][:,0:pos,:]
            values = self.values_cache[i][:,0:pos,:]
            new_key = x_K_projected[i]
            new_key = new_key.unsqueeze(1)
            new_value = x_V_projected[i]
            new_value = new_value.unsqueeze(1)
            keys = torch.cat((keys, new_key), dim=1)
            values = torch.cat((values, new_value), dim=1)
            attn_context = attention(x_Q_projected[i],keys,values,mask=mask)
            attn_contexts.append(attn_context)
        # concatenate all the attn_contexts for the mha heads
        attn_concatenated = torch.cat(attn_contexts,dim=2)
        out = self.final_linear(attn_concatenated)
        out = self.dropout(out)
        return out
















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
        K_projected = [f(key) for f in self.K_linears]
        V_projected = [f(value) for f in self.V_linears]
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
        return self.w_2(self.dropout(nn.functional.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self,
                emb_size: int,
                dropout: float,
                maxlen: int = 5000
        ):
        super(PositionalEncoding, self).__init__()
        embedding_matrix = torch.zeros(maxlen, emb_size)
        pos = torch.arange(0,maxlen).unsqueeze(1) # for broadcasting
        denominator = torch.exp(-torch.arange(0,emb_size,2)*math.log(10000)/emb_size)
        embedding_matrix[:, 0::2] = torch.sin(pos * denominator)
        embedding_matrix[:, 1::2] = torch.cos(pos * denominator)
        embedding_matrix = embedding_matrix.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', embedding_matrix)

    def forward(self, x):
        x = x + Variable(self.pos_embedding[:, x.size(1)], requires_grad=False)
        return self.dropout(x)

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
        return self.sublayer[1](x, lambda x: self.feed_forward(x))

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
    def __init__(self, size, self_attn, cached_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cached_attn = cached_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout), 3)
        self.size = size # d_model

    def forward(self, x, encoder_outputs, source_mask, target_mask, pos=None):
        m = encoder_outputs
        x = self.sublayer[0](x, lambda x: self.cached_attn(x,target_mask,pos=pos))
        x = self.sublayer[1](x, lambda x: self.self_attn(x,m,m,source_mask))
        return self.sublayer[2](x, lambda x: self.feed_forward(x))

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size) # d_model # is this necessary?

    def forward(self, x, encoder_outputs, source_mask, target_mask,pos=None):
        for layer in self.layers:
            x = layer(x, encoder_outputs, source_mask, target_mask,pos=pos)
        return self.norm(x)

"""
Takes in source, source_mask, target_mask
"""
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source),source_mask)

    def greedy_decode(self, encoder_outputs, source_mask, target, target_mask, pos):
        return self.decoder(self.tgt_embed(target.long()), encoder_outputs, source_mask, target_mask, pos)

    def decode(self, encoder_outputs, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target.long()), encoder_outputs, source_mask, target_mask)

    def forward(self, source, target, source_mask, target_mask):
        x = self.decode(self.encode(source,source_mask), source_mask, target, target_mask)
        return self.generator(x)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, max_seq_len=10):
    print("d_model: ", d_model)
    print("h: ", h)
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    # self,h,d_model,max_seq_len,dropout=0.1
    cached_attn = CachedMultiHeadedAttention(h, d_model, max_seq_len, dropout)
    # todo(annhe): instantiate Decoder 1st attention later with CachedMultiHeadedAttention
    # Question: how and when to clear the cache?
    ff = FeedForwardNetwork(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(cached_attn),c(ff),dropout),N),
        nn.Sequential(Embeddings(d_model,src_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,tgt_vocab),c(position)),
            Generator(d_model,tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg = torch.cat((self.trg, torch.zeros(trg.size(0),1)),dim=1)
            self.trg_y = trg[:, 1:]
            self.trg_y = torch.cat((torch.zeros(trg.size(0),1),self.trg_y),dim=1)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            future_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(V, batch_size, nbatches, d_model=10):
    "Generate random data for a src-tgt copy task."
    i = 0
    while i < nbatches:
        data = torch.from_numpy(np.random.randint(1, V, size=(batch_size, d_model)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
        i += 1

def attention_tests():
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

    # testing attention for source_masking

def main():
    vocab_size = 10
    vocab_size = 10
    model = make_model(10,10)



if __name__ == "__main__":
    main()
