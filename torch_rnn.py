from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def process_data(file_path):
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    # Read a file and split into lines
    def readLines(filename):
        lines = open(filename, encoding='utf=8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]


    for filename in findFiles(file_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    return category_lines, all_categories, n_categories

"""
Names into Tensors.

Names are represented as 2D matrices <line_length x 1 x n_letters>
"""

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_type='zeros'):
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_type = init_type

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, init_type='zeros'):
        if self.init_type == 'zeros':
            return torch.zeros(1, self.hidden_size)
        elif self.init_type == 'ones':
            return torch.ones(1, self.hidden_size)
        elif self.init_type == 'uniform':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.uniform_(tensor, a=0.0, b=1.0, generator=None)
            return tensor
        elif self.init_type == 'normal':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.normal_(tensor, mean=0.0, std=1.0, generator=None)
            return tensor
        elif self.init_type == 'glorot':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.xavier_uniform_(tensor, gain=1.0, generator=None)
            return tensor
        else:
            return torch.zeros(1, self.hidden_size)

class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM, self).__init__()
        self.cell_size = n_categories + input_size + hidden_size
        self.output_size = output_size
        self.cell = torch.zeros(1, self.cell_size)

        self.forget_gate = nn.Linear(self.cell_size, self.cell_size)
        self.input_gate = nn.Linear(self.cell_size, self.cell_size)
        self.candidate_gate = nn.Linear(self.cell_size, self.cell_size)
        self.output_gate = nn.Linear(self.cell_size, self.cell_size)
        self.o2o = nn.Linear(self.cell_size, self.output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = torch.tanh()

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        forget_layer = self.sigmoid(self.forget_gate(input_combined))
        input_layer = self.sigmoid(self.input_gate(input_combined))
        candidate_layer = self.tanh(self.candidate_gate(input_combined))
        self.cell = forget_layer * self.cell + input_layer * candidate_layer

        output_layer = self.sigmoid(self.output_gate(input_combined))
        hidden = output_layer * self.tanh(self.cell)

        # adding this
        output = self.o2o(hidden)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, cell, hidden

    def initHidden(self, init_type='zeros'):
        if self.init_type == 'zeros':
            return torch.zeros(1, self.hidden_size)
        elif self.init_type == 'ones':
            return torch.ones(1, self.hidden_size)
        elif self.init_type == 'uniform':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.uniform_(tensor, a=0.0, b=1.0, generator=None)
            return tensor
        elif self.init_type == 'normal':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.normal_(tensor, mean=0.0, std=1.0, generator=None)
            return tensor
        elif self.init_type == 'glorot':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.xavier_uniform_(tensor, gain=1.0, generator=None)
            return tensor
        else:
            return torch.zeros(1, self.hidden_size)

class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        output = self.i2o(input_combined)
        hidden = self.i2h(input_combined)
        out_combined = torch.cat((output,hidden),1)
        output = self.o2o(out_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, init_type='zeros'):
        if self.init_type == 'zeros':
            return torch.zeros(1, self.hidden_size)
        elif self.init_type == 'ones':
            return torch.ones(1, self.hidden_size)
        elif self.init_type == 'uniform':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.uniform_(tensor, a=0.0, b=1.0, generator=None)
            return tensor
        elif self.init_type == 'normal':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.normal_(tensor, mean=0.0, std=1.0, generator=None)
            return tensor
        elif self.init_type == 'glorot':
            tensor = torch.empty(1, self.hidden_size)
            torch.nn.init.xavier_uniform_(tensor, gain=1.0, generator=None)
            return tensor
        else:
            return torch.zeros(1, self.hidden_size)



# use Tensor.topk to get the index of the greatest value
def categoryFromOutput(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

# Get a random category and random line from that category
def randomTrainingPair(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

def randomTrainingExample(category_lines, all_categories):
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.fine(letter)] = 1
    return tensor

# ''LongTensor'' of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexs = [all_letters.find(line[li]) for li in range(1,len(line))]
    letter_index.append(nletters-1) # EOS
    return torch.LongTensor(letter_indexes)

# A helper function that samples (category,line) and transforms to
# (category, input, output)
def randomTrainingPairExample(category_lines, all_categories):
    category, line = randomTrainingPAir(category_lines, all_categories)
    category_tensor = categoryTensor(category)
    intput_tensor = inputTensor(line)
    target_tensor = targetTensor(line)
    return category_tensor, input_tensor, target_tensor


def train_gen_model(rnn, criterion, category_tensor, input_line_tensor, output_line_tensor, learning_rate, n_letters):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(input_line_tensor[i], hidden)

        loss += criterion(output, target_line_tensor[i])
    
    loss.backwards(retain_graph=True)

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size()[0]

def train_gen_rnn():
    rnn = RNN2(n_letters, 128, n_letters)

    n_iters = 1e5
    printer_every = 5e3
    all_losses = []
    total_loss = 0

    for iter in range(1, n_iters + 1):
        output, loss = train_gen_model(*randomTrainingPairExample())
        total_loss += loss_fn

        if iter % print_very == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

def sample(rnn, category, max_length = 20, start_letter='A'):
    with torch.no_grad(): 
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter
        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
    return output_name

def samples(rnn, category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

"""
Each loop of training will:
* Create input and target tensors
* Create a zeroed initial hidden state
* Read each letter in and,
* Keep a hidden state for next letter
* Compare final output to target
* Back-propagate
* Return the output and loss
"""
def train(rnn, criterion, category_tensor, line_tensor, learning_rate):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward(retain_graph=True)

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def train_categorical_rnn():
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()
    learning_rate = 0.005


if __name__ == "__main__":
    train_gen_rnn()
