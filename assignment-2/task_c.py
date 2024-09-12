import random

import torch
import matplotlib.pyplot as plt

x_train = torch.zeros((10000, 2), dtype=torch.float32)
y_train = torch.zeros((10000, 1), dtype=torch.float32)

index = 0
for i in range(100):
    for j in range(100):
        x1 = i / 100
        x2 = j / 100
        x_train[index] = torch.tensor([x1, x2])
        if not (x1 > 0.5 and x2 > 0.5) and not (x1 < 0.5 and x2 < 0.5):
            y_train[index] = 1.0
        index += 1


class XorGateSigmoidModel(torch.nn.Module):
    def __init__(self):
        super(XorGateSigmoidModel, self).__init__()

        # Store weights and biases in ParameterList
        self.W_list = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([[random.uniform(-1, 1)], [random.uniform(-1, 1)]])) for _ in range(4)])
        self.b_list = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([[random.uniform(-1, 1)]])) for _ in range(4)])

        # Output layer
        self.W_out = torch.nn.Parameter(torch.tensor([[random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]]))
        self.b_out = torch.nn.Parameter(torch.tensor([[random.uniform(-1, 1)]]))

        self.sigmoid = torch.nn.Sigmoid()

    def f(self, x):
        h_list = []
        for W, b in zip(self.W_list, self.b_list):
            h = self.sigmoid(x @ W + b)
            h_list.append(h)
        h = torch.cat(h_list, 1)
        return self.sigmoid(h @ self.W_out + self.b_out)

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


model = XorGateSigmoidModel()

epochs = 2000
loss_values = []

optimizer = torch.optim.Adam(model.parameters(), 0.1)
for epoch in range(epochs):
    loss_values.append(model.loss(x_train, y_train).item())
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print("epoch %d, loss %f" % (epoch, model.loss(x_train, y_train).item()))


print("loss = %s" % model.loss(x_train, y_train).item())

# Visualize result
fig = plt.figure("Xor Gate")
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

plt.plot(loss_values)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss over epochs')
plt.show()