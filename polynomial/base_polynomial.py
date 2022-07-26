from eye_utils import utils as util
import numpy as np
# t=(y-xw).T(y-xw)
# 0=2x.T*(y-xw)
# 2*x.T*x*w=2*x.T*y
# w=(x.T*x)^-1*x.T*y
# x=[a,b].T


class base_polynomial:
    def __int__(self,calibration_num):
        self._calib_num=calibration_num
    def get_matric(self,x,y):
       ...

    def do_estimation(self,point):
        ...