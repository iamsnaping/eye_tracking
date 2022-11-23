from polynomial.base_polynomial import base_polynomial
import numpy as np
from eye_utils import utils as util

from polynomial.base_polynomial import base_polynomial

#单眼
#坐标 向左 为x正方向 向下 为 y正方向
class pc_cr(base_polynomial):
    def __init__(self,calib_num):
        super(pc_cr, self).__init__(calib_num)
        self._w=[]
        for i in range(2):
            self._w.append(np.zeros((8,1)))
    def get_matric(self,x,y):
        pass
    #point-> pupil_center glint1 glitn2
    def do_estimation(self,points):
        gaze_point=np.zeros((2,1))
        glint1,glint2=self.get_vector(points)
        gaze_point+=self.data_rules(glint1.reshape(2,1))@self._w[0]
        gaze_point+=self.data_rules(glint2.reshape(2,1))@self._w[1]
        self._estimation=gaze_point/2
        return self._estimation
    def do_calibration(self,points_list,goals):
        list_len=len(points_list)
        kinds=len(points_list[0])
        x_l = [[]for i in range(kinds)]
        y_l = []
        for i in range(list_len):
            y_l.append(goals[i])
            for j in range(kinds):
                x_l[j].append(self.data_rules(points_list[i][j]))
        y=np.concatenate(y_l,axis=0)
        for j in range(kinds):
            self._w[j] += util.get_w(np.concatenate(x_l[j],axis=0), y).reshape((8, 1))

    #pupil glint1 glint2 shape->2
    def get_vector(self,points):
        glint1=points[0]-points[1]
        glint2=points[0]-points[2]
        glint1/=np.linalg.norm(glint1)
        glint2/=np.linalg.norm(glint2)
        return glint1,glint2

    def data_rules(self,point):
        x=np.zeros((1,8),dtype=np.float32)
        y=np.zeros((1,8),dtype=np.float32)
        x[0][0],x[0][1],x[0][2],x[0][3]=1.0,point[0,0],point[0,0]**3,point[1,0]**2
        y[0][4],y[0][5],y[0][6],y[0][7]=1.0,point[0,0],point[1,0],point[0,0]**2*point[1,0]
        return np.concatenate((x,y),axis=0)

