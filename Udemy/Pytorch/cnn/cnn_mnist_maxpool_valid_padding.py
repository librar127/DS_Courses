# ################################################################## #
# Import required libraries
# ################################################################## #
import torch
import torch.nn as nn
import torchvision.transforms as transforms 
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.optim import SGD

# ################################################################## #
# Load training and test datasets
# ################################################################## #
train_dataset = dsets.MNIST(root = '../data',
							transform = transforms.ToTensor(),
							train = True,
							download = True)

test_dataset = dsets.MNIST(root = '../data',
						   transform = transforms.ToTensor(),
						   train = False)

# ################################################################## #
# Make datasets Iterable
# ################################################################## #
batch_size = 100
num_epochs = 20

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
										   batch_size = batch_size,
										   shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
										   batch_size = batch_size,
										   shuffle = False)

# ################################################################## #
# Create Model Class
# ################################################################## #
class ConvolutionalNeuralNetModel(nn.Module):
	def __init__(self):
		super(ConvolutionalNeuralNetModel, self).__init__()

		# Convolution with Max pool 1
		self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 0)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

		# Convolution with Max pool 2
		self.cnn2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 0)
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

		# Fully connected 1 (readout)
		self.fc1 = nn.Linear(32 * 4 * 4, 10)


	def forward(self, x):

		# Convolution with Max pool 1
		out = self.cnn1(x)
		out = self.relu1(out)
		out = self.maxpool1(out)

		# Convolution with Max pool 2
		out = self.cnn2(out)
		out = self.relu2(out)
		out = self.maxpool2(out)


		# Fully connected 1 (readout)
		# Resize
		# Original size: (100, 32, 7, 7)
		# out.size(0): 100
		# New out size: (100, 32 *7 *7)
		out = out.view(out.size(0), -1)
		out = self.fc1(out)

		return out

# ################################################################## #
# Instantiate Model Class
# ################################################################## #
model = ConvolutionalNeuralNetModel()
if torch.cuda.is_available():
	model.cuda()

# ################################################################## #
# Instantiate loss Class
# ################################################################## #
criterion = nn.CrossEntropyLoss()

# ################################################################## #
# Instantiate optimizer Class
# ################################################################## #
learning_rate = 0.01
optimizer = SGD(model.parameters(), lr = learning_rate)

# ################################################################## #
# Parameters in depth
# ################################################################## #
print (model.parameters)
print len(list(model.parameters()))

# Convolution 1: 16 kernels
print list(model.parameters())[0].size()

# Convolution 1 bias: 16 kernels
print list(model.parameters())[1].size()

# Convolution 2: 32 kernels with depth = 16
print list(model.parameters())[2].size()

# Convolution 2 bias: 32 kernels with depth = 16
print list(model.parameters())[3].size()

# Fully connected layer 1
print list(model.parameters())[4].size()

# Fully connected layer 1 bias
print list(model.parameters())[5].size()

# ################################################################## #
# Train the model
# ################################################################## #

num_iter = 0
for epoch in range(num_epochs):
	
	loss = 0
	for i, (images, labels) in enumerate(train_loader):
		
		# 1. Convert inputs/labels to variables
		if torch.cuda.is_available():
			images = Variable(images.cuda())
			labels = Variable(labels.cuda())
		else:
			images = Variable(images)
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

		num_iter += 1

	# calculate accuracy
	correct = 0
	total = 0

	# Iterate through test dataset
	for images, labels in test_loader:

		# Convert inputs/labels to variables
		if torch.cuda.is_available():
			images = Variable(images.cuda())
		else:
			images = Variable(images)

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

	print "Epoch: {0: }, Iterations: {1:}, Loss: {2: .4f}., Accuracy: {3: .2f}".format(epoch+1, num_iter, loss.data[0], accuracy)
  