import deepxde as dde
import numpy as np
import torch

# Defining the Laplace equation for a disk
def pde(x, y):
    # Here, input x = [x0, x1] = [r, theta]
    # And output = y
    # i: which component of the output y you want
    # j: which component of the input x you want
    dy_r = dde.grad.jacobian(y, x, i=0, j=0)
    dy_rr = dde.grad.hessian(y, x, i=0, j=0)
    dy_thetatheta = dde.grad.hessian(y, x, i=1, j=1)
    
    # x[:, 0:1] extracts the r-coordinate values for all collocation points, shape (N,1)
    # x[:, 1:2] would be the θ-coordinate, shape (N,1)
    # x[:, 0] → gives you the first column flattened (shape (N,))
    return x[:, 0:1] * dy_r + x[:, 0:1]**2 *dy_rr + dy_thetatheta

def solution(x):
    r, theta = x[:, 0:1], x[:, 1:]
    return r * np.cos(theta)

# Use Rectangle if PDE is naturally expressed in (r, θ)
# Use Disk if PDE is defined in Cartesian (x, y)
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 2*np.pi])
# Here, x[:,0] = r ∈ [0,1]
# x[:,1] = θ ∈ [0, 2π]

bc_rad = dde.icbc.DirichletBC(geom, lambda x: np.cos(x[:, 1:2]),
                              lambda x, on_boundary: on_boundary and dde.utils.isclose(x[0], 1))
# Here, First lambda → gives the value of the boundary condition.
# Second lambda → selects the location where this BC applies.

data = dde.data.PDE(geom, pde, bc_rad, num_domain=2540, num_boundary=80, solution=solution)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

def feature_transform(x):
    return torch.cat(
        [x[:, 0:1] * torch.sin(x[:, 1:2]), x[:, 0:1] * torch.cos(x[:, 1:2])], dim=1
    )
# Here, x has shape (batch_size, 2) with columns [r, θ]
# feature_transform maps (r, θ) → (r*sinθ, r*cosθ)
# The transformed features are then fed into the neural network instead of raw (r, θ)
# Feeding raw (r, θ) can make it harder for the NN to learn smooth solutions near r=0
# The PDE residual is still computed w.r.t. the original coordinates (r, θ), so derivatives are handled correctly

net.apply_feature_transform(feature_transform)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
losshistory, train_state = model.train(iterations=20000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)