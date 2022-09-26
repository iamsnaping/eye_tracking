import numpy as np

from eye_utils import utils as util

class base_estimation(object):
    def __init__(self,w,h,d):
        self._w=w
        self._h=h
        self._d=d

    def calibration(self):
        pass
    def get_m_points(self):
        pass
    def gaze_estimation(self):
        pass

    # x_13 g0-m1
    # x_24 m0-g1
    # x_34 m1-g1
    # x_12 g0-m0
    # y_13 g1-m3
    # y_24 m2-g2
    # y_34 m3-g2
    # y_12 g1-m2
    # 34*12/13*24
    # 3->v 2->c
    def cross_ratio(self,g,m):
        x_13=g[0]-m[1]
        x_24=m[0]-g[1]
        x_34=g[1]-m[1]
        x_12=g[0]-m[0]
        y_13=g[1]-m[3]
        y_24=m[2]-g[2]
        y_34=m[3]-g[2]
        y_12=g[1]-m[2]
        # cross_ratio_x=util.get_cross(x_12,x_34)/util.get_cross(x_13,x_24)
        # cross_ratio_y=util.get_cross(y_12,y_34)/util.get_cross(y_13,y_24)
        cross_ratio_x = np.linalg.norm(x_12)*np.linalg.norm(x_34) / (np.linalg.norm(x_13)*np.linalg.norm(x_24))
        cross_ratio_y = np.linalg.norm(y_12)*np.linalg.norm(y_34) / (np.linalg.norm(y_13)*np.linalg.norm(y_24))
        return cross_ratio_x,cross_ratio_y

    def screen_cross(self,v,p):
        cross_ratio_x=p[0][0]/(self._w-p[0][0])
        cross_ratio_y=p[1][0]/(self._h-p[1][0])
        return cross_ratio_x,cross_ratio_y

