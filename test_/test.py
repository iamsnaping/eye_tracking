import math

import numpy as np
import torch
from sympy import symbols, Eq, solve, nsolve
from scipy.optimize import fsolve, root

# t1=torch.tensor([0,1],dtype=torch.float64).reshape((1,2))
# t2=torch.tensor([-1,-1],dtype=torch.float64).reshape((1,2))
# print(torch.multiply(t1,t2))

a=np.array([1,1,1]).reshape((3,1))
b=np.array([2,2,2]).reshape((3,1))
c=np.array([3,3,3]).reshape((3,1))
t=np.concatenate((np.concatenate((a,b,c),axis=1),np.array([0,0,0]).reshape((3,1))),axis=1)
print(t)
d=np.array([4,4,4,4]).reshape((1,4))
e=np.array([4,4,4,4]).reshape(4)
print(np.concatenate((t,d),axis=0))
k1,k2,k3=2
print(k1,k2,k3)