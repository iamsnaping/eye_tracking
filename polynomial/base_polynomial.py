from eye_utils import utils as util
import numpy as np
# t=(y-xw).T(y-xw)
# 0=2x.T*(y-xw)
# 2*x.T*x*w=2*x.T*y
# w=(x.T*x)^-1*x.T*y
# x=[a,b] y=[a,b].T


class base_polynomial:
    def __init__(self,calibration_num):
        self._calib_num=calibration_num
        self._estimation=np.zeros((2,1),dtype=np.float64)
    def get_matric(self,x,y):
        pass
    def data_rules(self,point):
        pass
    def do_calibration(self,points,goals):
        pass
    def do_estimation(self,point):
        pass
