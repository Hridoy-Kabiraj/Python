import deepxde as dde
import numpy as np
dde.config.backend = "pytorch"
dde.config.set_default_float("float64") # To match the input and output of the nn

# Define the poission equation
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    dy_yy = dde.grad.hessian(y, x, i=1, j=1)
    return -dy_xx - dy_yy - 1
# Here, x is your input variable (usually a 2D point [x0, x1] in DeepXDE for a 2D PDE)
# y is the neural network output at point x
# i=0, j=0 → second derivative of y w.r.t. the first input variable x[0] twice
# i=1, j=1 → second derivative of y w.r.t. the second input variable x[1]

def boundary(_, on_boundary):
    return on_boundary

geom = dde.geometry.Polygon([[0, 0], [1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]])
# Here, Each sublist [x, y] is a coordinate in 2D space
# If you plot these points in order and connect them, it will form a 6-sided polygon (hexagon-like but irregular). L-shaped

bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary)
data = dde.data.PDE(geom, pde, bc, num_domain=1200, num_boundary=120, num_test=1500)
net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
model.train(iterations=50000)
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)