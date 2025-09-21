import deepxde as dde 
import numpy as np
import torch

geom = dde.geometry.Interval(0, np.pi)

# Define sine function
sin = torch.sin

# Defining the poisson equation
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    summation = sum([i * sin(i * x) for i in range(1, 5)])
    return -dy_xx - summation - 8 * sin(8 * x)

# Defining the exact solution
def func(x):
    summation = sum([np.sin(i * x) / i for i in range(1, 5)])
    return x + summation + np.sin(8 * x) / 8

# Creating data for training
data = dde.data.PDE(geom, pde, [], num_domain=64, solution=func, num_test=400)

layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Instead of boundary condition we ar applying output transform to the nn. At x=0, u(x)=0 and at x=pi, u(x)=pi.
# This condition is satisfied by the below function
def output_transform(x, y):
    return x * (np.pi - x) * y + x

net.apply_output_transform(output_transform)

model = dde.Model(data, net)
model.compile("adam", lr=0.0001, decay=("inverse time", 1000, 0.3), metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations=30000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)


