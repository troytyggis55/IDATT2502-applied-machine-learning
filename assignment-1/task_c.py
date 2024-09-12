import matplotlib.pyplot as plt
import torch

with open("day_head_circumference.csv", "r") as file:
    file.readline()  # Skip the header
    lines = file.readlines()

x_train = torch.tensor([float(line.split(",")[0]) for line in lines]).reshape(-1, 1)
y_train = torch.tensor([float(line.split(",")[1]) for line in lines]).reshape(-1, 1)

class NonLinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid(x @ self.W + self.b) + 31  # Use the instantiated Sigmoid
        # function

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = NonLinearRegressionModel()

print("Initial loss: %s, W: %s, b: %s" % (model.loss(x_train, y_train), model.W, model.b))

optimizer = torch.optim.Adam([model.W, model.b], 0.01)
for epoch in range(1000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print("epoch %d, loss %f, W %f, b %f"
              % (epoch, model.loss(x_train, y_train).item(), model.W.item(), model.b.item()))

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
x = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(-1, 1)
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('day')
plt.ylabel('head circumference')
plt.plot(x, model.f(x).detach(), label='$f(x) = 20\sigma(xW+b)+31$')
plt.legend()
plt.show()

