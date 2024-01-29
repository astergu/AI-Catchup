import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

## Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # (input size, output size)
    
    def forward(self, x):
        out = self.linear(x)
        return out

# generate data
w = torch.tensor([2.0])
b = torch.tensor(3.5)
data_size = 10
inputs = torch.normal(0, 1, (data_size, len(w))).reshape(-1, 1)
target = torch.matmul(inputs, w) + b
target += torch.normal(0, 0.01, target.shape) # noise
target = target.reshape(-1, 1)

# main workflow
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# training
num_epochs = 200
for epoch in range(num_epochs):
    # clear gradient buffers
    optimizer.zero_grad()
    # get output from the model
    out = model(inputs)
    # backward pass
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss: {loss.item(): .3f}')


# evaluation
with torch.no_grad():
    predict = model(inputs).data.numpy()

print("--------------- Parameters learnt ---------------------")
for name, param in model.named_parameters():
    print(f"[{name}]: {param.data}")

fig = plt.figure(figsize=(10, 5))
plt.plot(inputs.numpy(), target.numpy(), 'ro', label='Original Data')
plt.plot(inputs.numpy(), predict, label='Fitting Line')
plt.legend()
plt.show()