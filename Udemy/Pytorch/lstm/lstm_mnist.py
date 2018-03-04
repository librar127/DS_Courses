import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root = '../data',
							train = True,
							transform = transforms.ToTensor(),
							download = True)

test_dataset = dsets.MNIST(root = '../data',
							train = False,
							transform = transforms.ToTensor())

print train_dataset.train_data.size(), train_dataset.train_labels.size(), test_dataset.test_data.size(), test_dataset.test_labels.size()


'''
STEP 2: MAKE DATASET ITERABLE
'''
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

'''
STEP 3: CREATE MODEL CLASS
'''
class LSTMModel(nn.Module):
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(LSTMModel, self).__init__()

		# Hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# Building RNN
		self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)

		# Readout layer
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):

		# initialize hidden state with zeros (layer_dim, batch_size, hidden_dim)
		if torch.cuda.is_available():
			h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
		else:
			h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		
		# initialize cell state
		if torch.cuda.is_available():
			c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
		else:
			c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		
		# 28 (image size) time step
		out, (hn, cn) = self.lstm(x, (h0, c0))

		# Index hidden state of last time step
		# out.size() --> 100, 28, 100 == out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
		out = self.fc(out[:, -1, :])

		return out

'''
STEP 4: INSTANTIATE MODEL CLASS
'''
input_dim = 28
hidden_dim = 100
layer_dim = 1 
output_dim = 10

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
if torch.cuda.is_available():
	model.cuda()

'''
STEP 5: INSTANTIATE LOSS CLASS
'''
creterion = nn.CrossEntropyLoss()

'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

'''
STEP 7: TRAIN MODEL
'''
iter = 0
seq_dim = 28
for epoch in range(num_epochs):
	for images, labels in train_loader:

		# Convert into torch Variables
		if torch.cuda.is_available():
			images = Variable(images.view(-1, seq_dim, input_dim).cuda())
			labels = Variable(labels.cuda())
		else:
			images = Variable(images.view(-1, seq_dim, input_dim))
			labels = Variable(labels)

		# Clear gradients w.r.t. parameters
		optimizer.zero_grad()

		# Forward pass to get output/Logits
		outputs = model(images)

		# Calculate the loss
		loss = creterion(outputs, labels)

		# Getting gradients w.r.t. parameters
		loss.backward()

		# Update parameters
		optimizer.step()

		iter += 1

		if iter % 500 == 0:

			# Calculate accuracy on test set
			correct = 0
			total = 0

			for images, labels in test_loader:

				# Convert into torch Variables
				if torch.cuda.is_available():
					images = Variable(images.view(-1, seq_dim, input_dim).cuda())
				else:
					images = Variable(images.view(-1, seq_dim, input_dim))

				# Forward pass to get outputs/logits
				outputs = model(images)

				# Get predicted labels 
				_, predicted = torch.max(outputs.data, 1)

				# Total number of labels
				total += labels.size(0)

				# Total correct prediction
				if torch.cuda.is_available():
					correct += (predicted.cpu() == labels).sum()
				else:
					correct += (predicted == labels).sum()

			accuracy = 100 * correct / total
			print "Iteration: {} Loss: {} Accuracy: {}".format(iter, loss.data[0], accuracy)