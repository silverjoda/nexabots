import torch as T
from torch.autograd import grad

a = T.tensor(2., requires_grad=True)
b = T.tensor(3., requires_grad=True)
c = T.pow(a, 5) * b

a_grad = grad(c, a, create_graph=True)
print(a_grad)
a_grad2 = grad(a_grad[0], a, create_graph=True)
print(a_grad2)
a_grad3 = grad(a_grad2[0], a, create_graph=True)
print(a_grad3)