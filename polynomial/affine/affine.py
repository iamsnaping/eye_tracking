from polynomial.base_polynomial import base_polynomial
import numpy as np
from eye_utils import utils as util

class affine(base_polynomial):
    def __int__(self,calib_num):
        super(affine, self).__int__(calibration_num=calib_num)

    def do_estimation(self, point):
        x = np.array([point[0][0], point[1][0], 1])
        return (x @ self._coefficient).T
    def get_matric(self,x,y):
        y=y.reshape((-1,1))
        mid=[]
        for i in range(len(x[0])):
            mid.append([x[0,i],x[1,i],1,0,0,0])
            mid.append([0,0,0,x[0,i],x[1,i],1])
        x=np.array(mid,dtype=np.float32).reshape((-1,6))
        self._coefficient= util.get_w(x,y)

