import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float().to(device)  # Move to GPU
y_train = torch.zeros((mnist_train.targets.shape[0], 10)).to(device)  # Create output tensor on GPU
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float().to(device)  # Move to GPU
y_test = torch.zeros((mnist_test.targets.shape[0], 10)).to(device)  # Create output tensor on GPU
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

batch_size = 64

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=3,
                          pin_memory=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=3,
                         pin_memory=True)



class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Updated model layers with increased depth and complexity
        self.logits = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean((self.f(x).argmax(1) == y).float())


# Instantiate model and move to GPU
model = ConvolutionalNeuralNetworkModel().to(device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Training loop
num_epochs = 20
accuracies = []

for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = model.loss(x_batch, y_batch)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Evaluation on the test set
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        total_samples = 0
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            accuracy = model.accuracy(x_batch, y_batch)
            batch_size = y_batch.size(0)
            total_accuracy += accuracy.item() * batch_size
            total_samples += batch_size

        epoch_accuracy = total_accuracy / total_samples
        accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {epoch_accuracy:.4f}")


# Function to display images and predictions
def show_images_and_predictions(model, dataset, num_images=10):
    model.eval()
    with torch.no_grad():
        for i in range(num_images):
            image, label = dataset[i]
            x = image.unsqueeze(0).to(device)
            output = model.f(x)
            prediction = output.argmax(1).item()

            plt.imshow(image.squeeze(0), cmap='gray')
            plt.title(f'Prediction: {prediction}, Actual: {label}')
            plt.show()

# Show images and predictions
show_images_and_predictions(model, mnist_test)
