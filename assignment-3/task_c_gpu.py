import torch
import torch.nn as nn
import torchvision

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)  # Move to GPU
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)  # Create output tensor on GPU
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)  # Move to GPU
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)  # Create output tensor on GPU
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)

class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # Each layer is scanned by a 5x5 kernel with padding 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Divide the size of the image by 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # Each layer is scanned by a 5x5 kernel with padding 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Divide the size of the image by 2
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout layer with 50% probability
            nn.Linear(1024, 10)
        )

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

# Instantiate model and move to GPU
model = ConvolutionalNeuralNetworkModel().to(device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)

accuracies = []

for epoch in range(20):
    for batch in range(len(x_train_batches)):
        # Forward pass and backpropagation
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b
        optimizer.zero_grad()  # Clear gradients for next step

    # Compute accuracy
    accuracies.append(model.accuracy(x_test, y_test))
    print(f"accuracy = {accuracies[-1]}")
