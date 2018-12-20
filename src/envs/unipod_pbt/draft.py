import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print(net)

MSE = nn.MSELoss()
optim = T.optim.Adam(net.parameters(), lr=1e-3)

X = T.rand(128, 32)
Y = T.rand(128, 10)

for i in range(10000):
    Y_ = net(X)

    optim.zero_grad()
    loss = MSE(Y, Y_)
    loss.backward()
    optim.step()

    if i % 100 == 0:
        print("Iters {}/{}, loss: {}".format(i, 10000, loss))

