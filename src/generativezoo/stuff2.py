import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
input = torch.randn(10, 256, requires_grad=True)
# target should be eye(10)
target = torch.eye(10)
optimizer.zero_grad()
output = model(input)
print(output)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)
loss.backward()
output = model(input.grad)
print(output)
print(target)