'''
STEPS:
1. Load dataset
2. Make dataset iterable
3. Create model class
4. Instantiate model class
5. Instantiate loss class
6. Instantiate optimizer class
7. Train model
'''

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# STEP 1: Load dataset
train_dataset = dsets.MNIST(root = '../data', 
							train=True,
							transform = transforms.ToTensor(),
							download = True)

test_dataset = dsets.MNIST(root = '../data',
						   train = False,
						   transform = transforms.ToTensor())

print train_dataset.train_data.size(), train_dataset.train_labels.size(), test_dataset.test_data.size(), test_dataset.test_labels.size()

# STEP 2: Make dataset Iterable
batch_size = 100
n_iters = 3000

num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
										   batch_size = batch_size,
										   shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
										  batch_size = batch_size,
										  shuffle = False)
# STEP 3: Create model class
class RNNModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()

		# Hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# Building RNN
		# batch_first = True- causes input/output tensors to be of shape (batch_size, seq_dim, input_dim)
		self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = True, nonlinearity = 'tanh')

		# Readout layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):

		# initialize hidden state with zeros
		# (layer_dim, batch_size, hidden_dim)
		h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

		# One time step
		out, hn = self.rnn(x, h0)

		# Index hidden state of last time step
		# out.size() --> 100, 28, 100
		# out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
		out = self.fc(out[:, -1, :])

		return out

# STEP 4: Instantiate model class
input_dim = 28
hidden_dim = 200
layer_dim = 2
output_dim = 10

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# Step 5: Instantiate loss class
criterion = nn.CrossEntropyLoss()

# Step 6: Instantiate optimizer class
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# parameters in depth
print "\nModel Parameters: ", len(list(model.parameters()))
'''
There are total 6 parameters
1. Input to Hidden layer [A1, B1]
2. Hidden layer to output linear function [A2, B2]
3. Hidden layer to Hidden layer linear function [A3, B3]
'''

print list(model.parameters())[0].size()
print list(model.parameters())[2].size()
print list(model.parameters())[1].size()
print list(model.parameters())[3].size()
print list(model.parameters())[4].size()
print list(model.parameters())[5].size()


# Step 7: Train model
seq_dim = 28 # Number of steps to unroll
iter = 0

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):

		# Load images as Torch Variable
		images = Variable(images.view(-1, seq_dim, input_dim))
		labels = Variable(labels)

		# Clear gradients w.r.t parameters
		optimizer.zero_grad()

		# Forward pass to get output/logits
		outputs = model(images)

		# Calculate loss : softmax -- > cross entropy loss
		loss = criterion(outputs, labels)

		# Getting gradients w.r.t. parameters
		loss.backward()

		# Updating parameters
		optimizer.step()

		iter += 1

		if iter % 500 == 0:
			 # Calculate Accuracy
			 correct = 0
			 total = 0

			 # Iterate through test dataset
			 for images, labels in test_loader:

				# Load images as Torch Variable
				images = Variable(images.view(-1, seq_dim, input_dim))

				# Forward pass only to get logits/output
				outputs = model(images)

				# Get predictions from the maximum value
				_, predicted = torch.max(outputs.data, 1)

				# Total number of labels
				total == labels.size(0)

				# Total correct predictions
				correct += (predicted == labels).sum()

			 accuracy = 100 * correct / total
			 # print loss
			 print "Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.data[0], accuracy)