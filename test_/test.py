import numpy as np
import torch
from sympy import symbols, Eq, solve, nsolve
from scipy.optimize import fsolve, root

# t1=torch.tensor([0,1],dtype=torch.float64).reshape((1,2))
# t2=torch.tensor([-1,-1],dtype=torch.float64).reshape((1,2))
# print(torch.multiply(t1,t2))


t=torch.tensor([1,2,3,4,5,6,7,8]).reshape((-1,2))
true_x=torch.tensor([20,20],dtype=torch.float64).reshape((1,2))
t1=torch.tensor([])

