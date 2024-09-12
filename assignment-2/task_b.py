import torch
import matplotlib.pyplot as plt

x_train = torch.empty((0, 2), dtype=torch.float32)
y_train = torch.empty((0, 1), dtype=torch.float32)

for i in range(100):
    for j in range(100):
        x1 = i / 100
        x2 = j / 100

        x_train = torch.cat((x_train, torch.tensor([[x1, x2]])), 0)
        if not (x1 > 0.5 and x2 > 0.5):
            y_train = torch.cat((y_train, torch.tensor([1.0]).reshape(-1, 1)), 0)
        else:
            y_train = torch.cat((y_train, torch.tensor([0.0]).reshape(-1, 1)), 0)


class NandGateSigmoidModel(torch.nn.Module):
    def __init__(self):
        super(NandGateSigmoidModel, self).__init__()  # Initialize the base class
        self.W = torch.nn.Parameter(torch.tensor([[0.0], [0.0]]))  # Register W as a parameter
        self.b = torch.nn.Parameter(torch.tensor([[0.0]]))  # Register b as a parameter
        self.sigmoid = torch.nn.Sigmoid()

    # Predictor
    def f(self, x):
        return self.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


model = NandGateSigmoidModel()


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
fig = plt.figure("Nand Gate")
ax = fig.add_subplot(projection='3d')
ax.plot(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(), 'o', label='$(x_1^{(i)}, x_2^{(i)}, y^{(i)})$', color='blue')

x1 = torch.linspace(torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), 100).reshape(-1, 1)
x2 = torch.linspace(torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), 100).reshape(-1, 1)
x1, x2 = torch.meshgrid(x1.squeeze(), x2.squeeze(), indexing='ij')

y = torch.zeros([100, 100])
for i in range(100):
    for j in range(100):
        y[i][j] = model.f(torch.tensor([[x1[i][j], x2[i][j]]])).detach()

ax.plot_wireframe(x1, x2, y, color='green')
ax.set_xlabel('$input1$')
ax.set_ylabel('$input2$')
ax.set_zlabel('$output$')
plt.legend()
plt.show()
