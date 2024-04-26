import torch.nn as nn
import torch
import numpy as np
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import math


class LSTM(nn.Module):
	def __init__(self, vocab_size, hidden_size):
		super(LSTM, self).__init__()
		self.lstm_cell = nn.LSTMCell(vocab_size, hidden_size)
		self.hidden = nn.zeros(1,hidden_size)
		self.cell_state = nn.zeros(1,hidden_size)
		# other init variables
		self.final_linear = nn.Linear(hidden_size, vocab_size)
		self.log_softmax = nn.LogSoftmax()

	def forward(self, x):
		# x is size N, L, D_vocab
		n = x.size(1)
		outputs = []
		for i in range(n):
			hidden, cell_state = self.lstm_cell(x[:,i,:], (hidden, cell_state))
			outputs.append(hidden[:,None,:])
		# want output of size N, L, D_hidden
		outputs = torch.cat(outputs, dim=1)
		outputs = self.final_linear(outputs)
		outputs = self.log_softmax(outputs,dim=2)
		return outputs

def train(model, criterion, optimizer, train_dataloader, vocab_size, hidden_size, n_epochs):
	for epoch in n_epochs:
		loss = 0
		for i, (x,y) in enumerate(train_dataloader): # make sure dataloader is shuffled
			# y = toTensorFormat(y)
			# x = toTensorFormat(x)
			# N x L x D
			optimizer.zero_grad()
			y_pred = model(x)
			loss += criterion(y_pred, y)
			loss.backward()
			optimizer.step()


def main():
	model = LSTM(vocab_size, hidden_size)
	criterion = nn.NLLLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


