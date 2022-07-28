import math
from eye_utils import utils as util
import numpy as np
import matplotlib.pyplot as plt
import torch
from sympy import symbols, Eq, solve, nsolve
from scipy.optimize import fsolve, root


# t=util.get_points_3d(np.array([1220.74563412,220.1347212,0.0]),
#                          np.array([272.74308375,273.8738555,0.0]),
#                          np.array([250.68277,259.503143,0.0]),
#                          np.array([251.993057,289.338593 ,0.0]))
# x=[272.74308375, 250.68277, 251.993057]
# y=[273.8738555 ,259.503143 ,289.338593]
# x.append(t[0])
# y.append(t[1])
# col=[1,2,3,4]
# plt.scatter(x,y,c=col)
# plt.show()
# a=np.array([1.0,1.0])
# b=np.array([2.0,2.0])
# print(np.cross(a,b))
# x=[358.58670952,364.757599,320.601532,358.59464719,353.32534764]
# y=[222.84529111,220.257584,221.87854,220.48382412,1788.10602238]
# col=[1,2,3,4,5]
# plt.scatter(x,y,c=col,s=10)
# plt.show()
a=np.array([1.0,-1.0])
b=np.array([-1.0,1.0])
print(np.cross(a,b))