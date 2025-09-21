import deepxde as dde
import numpy as np
import paddle  # MsFFN doesn't support PyTorch

A = 2
B = 50

# Define sin function
sin = paddle.sin

# Define poisson equation
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return dy_xx + (np.pi * A)**2 * sin(np.pi*A*x) + 0.1 * (np.pi * B)**2 * sin(np.pi*B*x)

# Define exact solution
def func(x):
    return np.sin(np.pi*A*x) + 0.1 * np.sin(np.pi*B*x)

geom = dde.geometry.Interval(-1, 1)
bc = dde.icbc.DirichletBC(geom, func, lambda _, on_boundary: on_boundary)
data = dde.data.PDE(geom, pde, bc, 1280, 2, train_distribution="pseudo", solution=func, num_test=10000)
# Here, The number 1280 is the number of training residual points sampled inside the domain
# And the number 2 is the number of training points sampled on the boundary

layer_size = [1] + [100] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.MsFFN(layer_size, activation, initializer, sigmas=[1, 10])
# Here, sigmas = The frequency/scale multipliers for the input, letting the network learn multiple scales of the solution
# one with the input as-is (σ = 1)
# and one with the input scaled up by 10 (σ = 10)
# This allows the network to capture both low-frequency (smooth) and high-frequency (oscillatory) behavior

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", 2000, 0.9))
# Here, decay=("inverse time", 2000, 0.9) means: start with lr = 0.001, and as training progresses,
# reduce it using an inverse time decay schedule that begins decaying around step 2000,
# with strength controlled by 0.9

pde_residual_resampler = dde.callbacks.PDEPointResampler(period=1)
# Here, PDEPointResampler resamples the collocation points inside the domain used for computing the PDE residual
# By resampling the points periodically, the network sees different points each time
# period=1 → resample every epoch

model.train(iterations=20000, callbacks=[pde_residual_resampler])

dde.saveplot(model.losshistory, model.train_state, issave=False, isplot=True)
