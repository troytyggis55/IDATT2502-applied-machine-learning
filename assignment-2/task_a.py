import torch
import matplotlib.pyplot as plt


inputs = torch.linspace(0, 1, 1000).reshape(-1, 1)
x_train = inputs
y_train = torch.tensor([1.0 if input < 0.5 else 0.0 for input in inputs]).reshape(-1, 1)

class NotGateSigmoidModel(torch.nn.Module):
    def __init__(self):
        super(NotGateSigmoidModel, self).__init__()  # Initialize the base class
        self.W = torch.nn.Parameter(torch.tensor([[0.0]]))  # Register W as a parameter
        self.b = torch.nn.Parameter(torch.tensor([[0.0]]))  # Register b as a parameter
        self.sigmoid = torch.nn.Sigmoid()

    # Predictor
    def f(self, x):
        return self.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


model = NotGateSigmoidModel()


optimizer = torch.optim.Adam([model.W, model.b], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        print("epoch %d, loss %f" % (epoch, model.loss(x_train, y_train).item()))



# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
x = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(-1, 1)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, model.f(x).detach(), label='$f(x) = relu(xW+b)$')
plt.legend()
plt.show()