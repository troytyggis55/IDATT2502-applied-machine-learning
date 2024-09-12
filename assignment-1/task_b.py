import torch
import matplotlib.pyplot as plt

file = open("day_length_weight.csv", "r")

# Read the first line of the file
file.readline()
line = file.readline()


day_train = torch.tensor([]) # y
lenght_weight_train = torch.tensor([])  # x1 and x2


# Read the rest of the lines
while line:
    values = line.split(",")

    day_train = torch.cat((day_train, torch.tensor([float(values[0])])), 0)

    lenght_weight_tensor = torch.tensor([float(values[1]), float(values[2])])
    lenght_weight_train = torch.cat((lenght_weight_train, lenght_weight_tensor), 0)

    line = file.readline()
file.close()

print("Values fetched from file")

day_train = day_train.reshape(-1, 1)
lenght_weight_train = lenght_weight_train.reshape(-1, 2)

print("Data reshaped")

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables
        # calculation
        # of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

epocs = 25000
lr = 0.1

optimizer = torch.optim.Adam([model.W, model.b], lr)
for epoch in range(epocs):
    model.loss(lenght_weight_train, day_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

    if epoch % 1000 == 0:
        print("Epoch: %d, Loss: %s" % (epoch, model.loss(lenght_weight_train, day_train).item()))

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(lenght_weight_train, day_train)))

# Visualize result in 3D
fig = plt.figure("Epochs: " + str(epocs) + ", Learning rate: " + str(lr))
ax = fig.add_subplot(projection='3d')
ax.plot(lenght_weight_train[:, 0].squeeze(), lenght_weight_train[:, 1].squeeze(), day_train[:, 0].squeeze(), 'o', label='$(x_1^{(i)}, x_2^{(i)}, y^{(i)})$', color='blue')

length_range = torch.arange(torch.min(lenght_weight_train[:, 0]), torch.max(lenght_weight_train[:, 0]), 1)
weight_range = torch.arange(torch.min(lenght_weight_train[:, 1]), torch.max(lenght_weight_train[:, 1]), 1)

length_grid, weight_grid = torch.meshgrid(length_range, weight_range)

grid_points = torch.cat((length_grid.reshape(-1, 1), weight_grid.reshape(-1, 1)), 1)

predicted_days = model.f(grid_points).detach().reshape(length_grid.shape)

# Plot the wireframe
ax.plot_wireframe(length_grid, weight_grid, predicted_days, color='green', label='$f(\\mathbf{x}) = xW+b$')
ax.set_xlabel('$length$')
ax.set_ylabel('$weight$')
ax.set_zlabel('$day$')

plt.legend()
plt.show()
