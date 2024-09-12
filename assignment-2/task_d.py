import torch
import torchvision
import matplotlib.pyplot as plt
import sys
import select

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

class SoftmaxModel(torch.nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        #self.hidden_layer = torch.nn.Linear(784, 256)  # Hidden layer with 128 units
        #self.output_layer = torch.nn.Linear(256, 10)  # Output layer
        #self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.layer = torch.nn.Linear(784, 10)

    def forward(self, x):
        #x = self.relu(self.hidden_layer(x))  # Apply ReLU activation to hidden layer
        #x = self.softmax(self.output_layer(x))  # Apply softmax to output layer
        return self.softmax(self.layer(x))
        #return self.softmax(self.output_layer(self.relu(self.hidden_layer(x))))

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.forward(x), y)


model = SoftmaxModel()

optimizer = torch.optim.Adam(model.parameters(), 0.01)

epoch = 0
test_accuracy = (model.forward(x_test).argmax(1) == y_test.argmax(1)).float().mean().item()

while test_accuracy < 0.9:
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    test_accuracy = (model.forward(x_test).argmax(1) == y_test.argmax(1)).float().mean().item()

    if epoch % 40 == 0:
        print("epoch %d, loss %f, accuracy %f" % (epoch, model.loss(x_train, y_train).item(),
                                                  test_accuracy))

    # Check for "end" input from the terminal
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        if sys.stdin.readline().strip() == "end":
            print("Ending training loop as 'end' was entered in the terminal.")
            break

    epoch += 1


weights = model.layer.weight.detach().reshape(10, 28, 28)

# Visualize the first 10 images
fig, ax = plt.subplots(1, 10)
for i in range(min(10, weights.shape[0])):
    ax[i].imshow(weights[i], cmap='gray')
    ax[i].axis('off')
plt.suptitle('Weight visualization')
plt.show()

