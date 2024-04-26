import torch.nn as nn
import torch
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import math

def clones(module, N):
    "Produces N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LSTM(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(LSTM, self).__init__()
		self.lstm_cell = nn.LSTMCell(vocab_size, hidden_size)
		self.hidden = nn.zeros(1,hidden_size)
		self.cell_state = nn.zeros(1,hidden_size)
		# other init variables

	def forward(self, x):
		# x is size N, L, D_vocab
		n = x.size(1)
		outputs = []
		for i in range(n):
			hidden, cell_state = self.lstm_cell(x[:,i,:], (hidden, cell_state))
			outputs.append(hidden[:,None,:])
		# want output of size N, L, D_hidden
		outputs = torch.cat(outputs, dim=1)
