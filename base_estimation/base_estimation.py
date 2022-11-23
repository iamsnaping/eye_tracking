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


    # def cross_ratio(self,g,m):
    #     x_13=g[0]-m[1]
    #     x_24=m[0]-g[1]
    #     x_34=g[1]-m[1]
    #     x_12=g[0]-m[0]
    #     y_13=g[1]-m[3]
    #     y_24=m[2]-g[2]
    #     y_34=m[3]-g[2]
    #     y_12=g[1]-m[2]
    #     # cross_ratio_x=util.get_cross(x_12,x_34)/util.get_cross(x_13,x_24)
    #     # cross_ratio_y=util.get_cross(y_12,y_34)/util.get_cross(y_13,y_24)
    #     cross_ratio_x = np.linalg.norm(x_12)*np.linalg.norm(x_34) / (np.linalg.norm(x_13)*np.linalg.norm(x_24))
    #     cross_ratio_y = np.linalg.norm(y_12)*np.linalg.norm(y_34) / (np.linalg.norm(y_13)*np.linalg.norm(y_24))
    #     return cross_ratio_x,cross_ratio_y


# g1 m1(x) m0(w/2) g0(w)
# ((w-w/2)*x)/((w-x)w/2)
# g2
# m2(y)
# m3(h/2)
# g3(h)

    #g0-m0 * m1-g1 /g0-m1 * m0-g1
    #g3-m3*m2-g2/g3-m2*m3-g2
    def cross_ratio(self,g,m):
        # print(g)
        # print(m)
        x_1=g[0]-m[0]
        x_2=m[1]-g[1]
        x_3=g[0]-m[1]
        x_4=m[0]-g[1]
        y_1=g[3]-m[3]
        y_2=m[2]-g[2]
        y_3=g[3]-m[2]
        y_4=m[3]-g[2]
        x1=(g[0]-g[1])/np.linalg.norm(g[0]-g[1])
        x2=x_3/np.linalg.norm(x_3)
        x3=(g[3]-g[2])/np.linalg.norm(g[3]-g[2])
        x4=y_3/np.linalg.norm(y_3)
        #w-x
        cosa = (x1 @ x2) / ((np.linalg.norm(x1) * np.linalg.norm(x2)))
        cosb = (x3 @ x4) / ((np.linalg.norm(x3) * np.linalg.norm(x4)))
        x2 = x_2 / np.linalg.norm(x_2)
        x4 = y_2 / np.linalg.norm(y_2)
        #x
        cosc = (x1 @ x2) / ((np.linalg.norm(x1) * np.linalg.norm(x2)))
        cosd = (x3 @ x4) / ((np.linalg.norm(x3) * np.linalg.norm(x4)))
        cross_ratio_x = np.linalg.norm(x_1)*np.linalg.norm(x_2) / (np.linalg.norm(x_3)*np.linalg.norm(x_4))
        cross_ratio_y = np.linalg.norm(y_1)*np.linalg.norm(y_2) / (np.linalg.norm(y_3)*np.linalg.norm(y_4))
        # for _ in g:
        #     print(_)
        # print(m)
        # print(x2,x1,cosc,cosa,cosd,cosb)
        # print(cross_ratio_x,cross_ratio_y,end=' ')
        if cosc<0:
            cross_ratio_x=-(cross_ratio_x)/(1-cross_ratio_x)
        elif cosa<0:
            cross_ratio_x =cross_ratio_x/(cross_ratio_x-1)
        else:
            cross_ratio_x=cross_ratio_x/(1+cross_ratio_x)

        if cosd < 0:
            cross_ratio_y = -(cross_ratio_y) / (1 - cross_ratio_y)
        elif cosb < 0:
            cross_ratio_y = cross_ratio_y / (cross_ratio_y - 1)
        else:
            cross_ratio_y = cross_ratio_y / (1 + cross_ratio_y)

        # print(cross_ratio_x,cross_ratio_y)
        return cross_ratio_x,cross_ratio_y


    def screen_cross(self,v,p):
        cross_ratio_x=p[0][0]/(self._w-p[0][0])
        cross_ratio_y=p[1][0]/(self._h-p[1][0])
        return cross_ratio_x,cross_ratio_y

