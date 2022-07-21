from eye_utils import utils as utils

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
    def cross_ratio(self,v,m):
        cross_ratio_x = ((v[0][0] * m[1][0] - m[0][0] * v[1][0]) * (m[0][1] * v[1][1] - v[0][1] * m[1][1])) \
                        / ((v[0][0] * m[1][1] - m[0][1] * v[1][0]) * (m[0][0] * v[1][1] - v[0][1] * m[1][0]))
        cross_ratio_y = ((v[0][1] * m[1][2] - m[0][2] * v[1][1]) * (m[0][3] * v[1][2] - v[0][2] * m[1][3])) / \
                        ((v[0][1] * m[1][3] - m[0][3] * v[1][1]) * (m[0][2] * v[1][2] - v[1][2] * m[1][2]))
        return cross_ratio_x,cross_ratio_y
