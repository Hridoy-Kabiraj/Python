import deepxde as dde
import numpy as np
import torch

n = 2
k0 = 2*np.pi*n # wavenumber
wave_len = 1 / n  # wave length
precision_train = 10 # 10 points per wavelength for the training
precision_test = 30
hard_constraint = True
weights = 100 # if hard_constraint == False


# Helmholtz equation
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    f = k0**2*torch.sin(k0*x[:, 0:1])*torch.sin(k0*x[:, 1:2])  # Source term

    return -dy_xx -dy_yy - k0*y - f

# Exact solution
def solution(x):
    return np.sin(k0*x[:, 0:1])*np.sin(k0*x[:, 1:2])

def transform(x, y):
    res = x[:, 0:1] * (1 - x[:, 0:1]) *x[:, 1:2] *(1 - x[:, 1:2])
    return res * y

def on_boundary(_, on_boundary):
    return on_boundary

geom = dde.geometry.Rectangle([0, 0], [1, 1])
hx_train = wave_len / precision_train # (grid spacing) x_train is the distance between two consecutive training sample points
nx_train = int(1 /hx_train) # (number of grid points) nx is the number of discrete points along one dimension of the domain

hx_test = wave_len / precision_test
nx_test = int(1 / hx_test)

if hard_constraint == True:
    bc = []
else:
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, on_boundary)

data = dde.data.PDE(geom, pde, bc, num_domain=nx_train**2, num_boundary=4*nx_train, solution=solution, num_test=nx_test**2)

net = dde.nn.FNN([2] + [150] * 3 + [1], "sin", "Glorot uniform")

if hard_constraint == True:
    net.apply_output_transform(transform)

model = dde.Model(data, net)

if hard_constraint == True:
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
else:
    loss_weight = [1, weights]
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], loss_weights=loss_weight)

losshistory, train_state = model.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)



