import deepxde as dde
import numpy as np

# defining derivatives for right boundary
def ddy(x, y):
    return dde.grad.hessian(y, x)

def dddy(x, y):
    return dde.grad.jacobian(ddy(x, y), x)

# Defining the Euler beam problem
def pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    dy_xxxx = dde.grad.hessian(dy_xx, x)

    return dy_xxxx + 1

def boundary_l(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def boundary_r(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)

# Defining the exact solution
def soultion(x):
    return - (x**4) / 24 + x**3 / 6 - x**2 / 4

geom = dde.geometry.Interval(0, 1)

bc1 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_l)
bc2 = dde.icbc.NeumannBC(geom, lambda x: 0, boundary_l)
bc3 = dde.icbc.OperatorBC(geom, lambda x, y, _: ddy(x, y), boundary_r)
bc4 = dde.icbc.OperatorBC(geom, lambda x, y, _: dddy(x, y), boundary_r)
# Here, ddy(x, y) is derivative function that only takes x and y
# OperatorBC will try to call it as operator(x, y, params) (3 arguments)
# So wrap it in a lambda that ignores the third argument _

data = dde.data.PDE(
    geom, pde, [bc1, bc2, bc3, bc4],
    num_domain=10, num_boundary=2,
    solution=soultion,
    num_test=100
)

net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)