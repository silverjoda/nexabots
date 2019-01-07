import numpy as np
import torch as T

x1 = T.tensor(2., requires_grad=True)
x2 = T.tensor(1., requires_grad=True)
w1 = T.tensor(np.pi/2., requires_grad=True)
w2 = T.tensor(np.pi, requires_grad=True)
b = T.tensor(0.)
l = T.tensor(2.)

y = T.sin(x1*w1 + x2*w2) + b
L = (y-l).pow(2)
L.backward()
print(y)
print(L)
print(w1.grad, x1.grad)
