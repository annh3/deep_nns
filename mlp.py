import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module): # in pytorch, networks always subclass nn.Module
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	# set the mdoel to training mode for batch normalization and dropout layers

	model.train()
	for batch, (X, y) in enumerate(dataloader):
		pred = model(X)
		loss = loss_fn(pred, y)

		# Backpropagation
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * 64+ len(X)
			print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
	# Set the model to evaluation mode
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	test_loss /= num_batches
	correct /= size 
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
	device = (
		"cuda"
		if torch.cuda.is_available()
		else "mps"
		if torch.backends.mps.is_available()
		else "cpu"
	)

	model = NeuralNetwork().to(device)
	print(model)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

	epochs = 10
	for t in range(epochs):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loop(train_dataloader, model, loss_fn, optimizer)
		test_loop(test_dataloader, model, loss_fn)
	print("Done!")


def practice():
	device = (
		"cuda"
		if torch.cuda.is_available()
		else "mps"
		if torch.backends.mps.is_available()
		else "cpu"
	)

	model = NeuralNetwork().to(device)
	print(model)

	X = torch.rand(1, 28, 28, device=device)
	logits = model(X)
	pred_probab = nn.Softmax(dim=1)(logits)
	y_pred = pred_probab.argmax(1)
	print(f"Predicted class: {y_pred}")

	# mini-batch of size 3
	input_image = torch.rand(3,28,28)
	print(input_image.size())

	flatten = nn.Flatten()
	flat_image = flatten(input_image)
	print(flat_image.size())

	# model parameters
	print(f"Model structure: {model}\n\n")

	for name, param in model.named_parameters():
		print(f"Layer: {name} | Size: {param.size()} | Values {param[:2]}")

if __name__ == "__main__":
	main()