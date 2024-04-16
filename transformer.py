from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, line_tensor

from torch import TensorDataset
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
	language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

	for data_sample in data_iter:
		yield token_transform[language](data_sample[language_index[language]])

	# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
	# Training data Iterator
	train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
	# Create torchtext's Vocab object
	vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
		min_freq=1,
		specials=special_symbols,
		special_first=True
		)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
	vocab_transform[ln].set_default_index(UNK_IDX)


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


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
	def __init__(self, vocab_size: int, emb_size):
		super(TokenEmbedding, self).__init__()
		self.embedding = nn.Embedding(vocab_size, emb_size)
		self.emb_size = emb_size

	def forward(self, tokens: Tensor):
		return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


