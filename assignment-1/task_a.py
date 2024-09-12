import torch
import matplotlib.pyplot as plt


file = open("length_weight.csv", "r")

# Read the first line of the file
file.readline()
line = file.readline()

lenght_train = torch.tensor([])
weight_train = torch.tensor([])

# Read the rest of the lines
while line:
    values = line.split(",")

    lenght_train = torch.cat((lenght_train, torch.tensor([float(values[0])])), 0)
    weight_train = torch.cat((weight_train, torch.tensor([float(values[1])])), 0)

    line = file.readline()
file.close()

lenght_train = lenght_train.reshape(-1, 1)
weight_train = weight_train.reshape(-1, 1)

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation
        # of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        #return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(
        # self.f(x), y) to possibly increase numberical stability
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# To make sure the model does not diverge, decrease the learning rate, and compensate by increasing the number of epochs
#optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
optimizer = torch.optim.Adam([model.W, model.b], 0.1)

for epoch in range(1000):
    model.loss(lenght_train, weight_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(lenght_train, weight_train)))

# Visualize result
plt.plot(lenght_train, weight_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('length')
plt.ylabel('weight')
x = torch.tensor([[torch.min(lenght_train)], [torch.max(lenght_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()
