import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure

import ga

# Cost test function
def sphere(x):
    return np.sum(x**2)


# Problem Definition
problem = structure()
problem.costfunc = sphere
problem.nvar = 5
problem.varmin = -10
problem.varmax = 10


# GA parameters
params = structure()
params.maxit = 100
params.npop = 1000
params.pc = 1
params.gamma = 0.1
params.mu = 0.1
params.sigma = 0.1
params.beta = 1


# RunGA

out = ga.run(problem, params)


# Results
#plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel("Iterations")
plt.ylabel("Best cost")
plt.title("Genetic Algorithm (GA)")
plt.grid(True)
plt.show()

