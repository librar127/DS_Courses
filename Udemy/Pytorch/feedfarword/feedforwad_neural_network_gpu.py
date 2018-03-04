# Import required libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.optim import SGD


# Step 1: Load MNIST dataset from torch
train_dataset = dsets.MNIST(root = './data', 
							train = True, 
							transform = transforms.ToTensor(), 
							download = True)

test_dataset = dsets.MNIST(root = './data', 
						  train = False, 
						  transform = transforms.ToTensor())

# Step 2: Make dataset Iterable
batch_size = 100
n_iter = 3000
num_epochs = n_iter / (len(train_dataset) / batch_size) # TOB: provide num_iter as input and print accuracy/loss at ech iteration
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
										   batch_size = batch_size,
										   shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
	  									  batch_size = batch_size,
	  									  shuffle = False)

# Step 3: Create Model class
class FeedForwardNeuralNetModel(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(FeedForwardNeuralNetModel, self).__init__()

		# Linear function 1: 784 --> 100
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		# Non-Linearity 1
		self.relu1 = nn.ReLU()

		# Linear function 2: 100 --> 100
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		# Non-Linearity 2
		self.relu2 = nn.ReLU()

		# Linear function 3: 100 --> 100
		self.fc3 = nn.Linear(hidden_dim, hidden_dim)
		# Non-Linearity 3
		self.relu3 = nn.ReLU()

		# Linear Function 4 (readout): 100 --> 10
		self.fc4 = nn.Linear(hidden_dim, output_dim)


	def forward(self, x):

		# Linear function 1 
		out = self.fc1(x)
		# Non-Linearilty 1 
		out = self.relu1(out)

		# Linear function 2
		out = self.fc2(out)
		# Non-Linearilty 2 
		out = self.relu2(out)

		# Linear function 3
		out = self.fc3(out)
		# Non-Linearilty 3 
		out = self.relu3(out)

		# Linear function 4 (readout)
		out = self.fc4(out)

		return out

# Step 4: Instantiate model class
input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedForwardNeuralNetModel(input_dim, hidden_dim, output_dim)
if torch.cuda.is_available():
	model.cuda()
	

# Step 5: Instantiate optimizer class
# Feedforward neural Network: Cross Entropy Loss
# Logistic Regression: Cross Entropy Loss
# Linear Regression: MSE
criterion = nn.CrossEntropyLoss()

# Step 5: Instantiate optimizer class
# theta = theta - learning_rate * parameters (gradients)
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

'''
# Parameters in depth
print (model.parameters())
print (len(list(model.parameters())))

# Hidden layer parameters
print (list(model.parameters())[0].size())

# FC1 bias parameters
print (list(model.parameters())[1].size())

# FC2 parameters
print (list(model.parameters())[2].size())

# FC2 bias parameters
print (list(model.parameters())[3].size())
'''

iter = 0
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		
		# 1. Convert inputs/labels to variables
		if torch.cuda.is_available():
			images = Variable(images.view(-1, 28*28).cuda())
			labels = Variable(labels.cuda())
		else:
			images = Variable(images.view(-1, 28*28).cuda())
			labels = Variable(labels)
			
		# 2. Clear gradient buffers
		optimizer.zero_grad()

		# 3. Get output given input
		outputs = model(images)

		# 4. Get loss -----> softmax entropy loss
		loss = criterion(outputs, labels)

		# 5. Get gradients w.r.t. parameters
		loss.backward()

		# 6. Update parameters using gradients
		optimizer.step()


		iter += 1

		# Print loss with every 500 iterations on test dataset
		if iter % 500 == 0:

			# calculate accuracy
			correct = 0
			total = 0

			# Iterate through test dataset
			for images, labels in test_loader:

				# Convert inputs/labels to variables
				if torch.cuda.is_available():
					images = Variable(images.view(-1, 28*28).cuda())
				else:
					images = Variable(images.view(-1, 28*28))

				# Get output given input
				outputs = model(images)

				# Get predictions from the maximum value
				_, predicted = torch.max(outputs.data, 1)

				# Get total number of labels on test prediction
				total += labels.size(0)

				# Total number of correct prediction
				if torch.cuda.is_available():
					correct += (predicted.cpu() == labels).sum()
				else:
					correct += (predicted == labels).sum()

			accuracy = 100 * correct / total*1.

			print "Iterations: {0:}, Loss: {1: .4f}., Accuracy: {2: .2f}".format(iter, loss.data[0], accuracy)