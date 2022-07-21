from base_estimation.base_estimation import base_estimation
import numpy as np
import torch
from eye_utils import utils as utils

class crdd(base_estimation):
    def __init__(self,w,h,d):
        super(crdd,self).__init__(w,h,d)
        self._points=torch.zeros((2,4),dtype=torch.float64)
        self._points_center=torch.zeros((2,1),dtype=torch.float64)
        self._m_points=torch.zeros((2,4),dtype=torch.float64)
        self._virtual_points=torch.zeros((2,5),dtype=torch.float64)
        self._pupil_center=torch.zeros((1,2),dtype=torch.float64)
        self._optical=torch.zeros((1,2),dtype=torch.float64)
        self._fifth=np.zeros((2,1),dtype=np.float64)
        self._vanishing_points=torch.zeros((2,2),dtype=torch.float64)
        self._alpha=torch.tensor([[2.0]],dtype=torch.float64,requires_grad=True)
        self._gaze_estimation=torch.zeros((2,1),dtype=torch.float64)
        self._ma=torch.zeros((2,1),dtype=torch.float64)
        self._z0=1.0


    def calibration(self,true_x,glints):
        glints=torch.tensor(glints,dtype=torch.float64)
        mean=torch.zeros((2,1),dtype=torch.float64)
        es=torch.zeros((4,2),dtype=torch.float64)
        for j in range(2):
            for i in range(len(true_x)):
                x=torch.tensor(glints[i][0,0:4],dtype=torch.float64).reshape((4,1))
                y=torch.tensor(glints[i][1,0:4],dtype=torch.float64).reshape((4,1))
                x=torch.matmul(x-glints[i][4][0],self._alpha).reshape((4,1))+glints[i][4][0]
                y=torch.matmul(y-glints[i][4][1],self._alpha).reshape((4,1))+glints[i][4][1]
                p2 = torch.concat([x, y], dim=1)
                v1 = utils.get_points([p2[0].reshape((2, 1)), p2[1].reshape((2, 1))],
                                [p2[3].reshape((2, 1)), p2[2].reshape((2, 1))])
                v2 = utils.get_points([p2[2].reshape((2, 1)), p2[1].reshape((2, 1))],
                                [p2[3].reshape((2, 1)), p2[0].reshape((2, 1))])
                m1 = utils.get_points([v1, p2[1].reshape((2, 1))], [v2, self._points_center])
                m2 = utils.get_points([v1, p2[1].reshape((2, 1))], [v2, self._pupil_center])
                m3 = utils.get_points([v2, p2[1].reshape((2, 1))], [v1, self._points_center])
                m4 = utils.get_points([v2, p2[1].reshape((2, 1))], [v1, self._points_center])
                cross_x = ((p2[0][0] * m1[1] - p2[0][1] * m1[0]) * (m2[0] * p2[1][1] - p2[1][0] * m2[1])) / (
                            (p2[0][0] * m2[1] - m2[0] * p2[0][1]) * (m1[0] * p2[1][1] - p2[1][0] * m1[1]))
                x = (10.0 * cross_x) / (1 + cross_x)
                cross_y = ((p2[1][0] * m3[1] - m3[0] * p2[1][1]) * (m4[0] * p2[2][1] - p2[2][0] * m4[1])) / (
                            (p2[1][0] * m4[1] - m4[0] * p2[1][1]) * (m3[0] * p2[2][1] - m3[1] * p2[2][0]))
                y = (10 * cross_y) / (1 + cross_y)
                es[i]=true_x[i]-torch.concat([x,y],dim=0).reshape((1,2))
                mean+=es[i]
            mean/=4
            loss=mean.norm()
            loss-=mean.norm()
            for i in range(4):
                loss+=(es[i]-mean).norm()
            loss.backward()
            with torch.no_grad:
                self._alpha-=0.1*self._alpha.grad
            self._alpha.grad.zero_()
        self._ma=mean.reshape((2,1))

    def gaze_estimation(self,zt):
        v=self._virtual_points
        m=self._m_points
        tu=self.cross_ratio(v,m)
        cross_ratio_x,cross_ratio_y=tu[0],tu[1]
        self._gaze_estimation[0][0]=(self._w*cross_ratio_x)/(1.0+cross_ratio_x)
        self._gaze_estimation[1][0]=(self._h*cross_ratio_y)/(1.0+cross_ratio_y)
        self._gaze_estimation+=self._ma*zt/self._z0
        return self._gaze_estimation
    def get_virtual_points(self):
        fifth=self._points[:,4]
        self._virtual_points-=fifth
        self._virtual_points*=self._alpha
        self._virtual_points+=self._fifth
    def get_m_points(self):
        for i in range(2):
            self._m_points[0+i*3]=utils.get_points([self._vanishing_points[::,0^i],self._virtual_points[::,2]],
                                               [self._vanishing_points[::,1^i],self._points_center])
            self._m_points[1+i*1]=utils.get_points([self._vanishing_points[::,0^i],self._virtual_points[::,2]],
                                               [self._vanishing_points[::,1^i],self._pupil_center])

    def get_vanish_points(self):
        v=self._virtual_points
        for i in range(2):
            self._vanishing_points[i]=utils.get_points([v[::,0],v[::,1+i*2]],
                                                    [v[::,3-i*2],v[::,2]])
            # flag=0
            # if self._virtual_points[0+i*2][1^i]==self._virtual_points[1][1^i]:
            #     self._vanishing_points[i][1^i]=self._virtual_points[1][1^i]
            #     flag+=1
            # if self._virtual_points[2-i*2][1^i]==self._virtual_points[3][1^i]:
            #     if self._vanishing_points[i][1^i]==self._virtual_points[1][1^i]:
            #         self._m_points[2-i*1][1^i]=self._pupil_center[1^i]
            #         self._m_points[3-i*3][1^i]=self._points_center[1^i]
            #     else:
            #         self._vanishing_points[i][1^i]=self._virtual_points[3][1^i]
            #     flag+=2
            # if flag==2:
            #     self._vanishing_points[0][0]=self._virtual_points[0][0]+\
            #                                  (self._vanishing_points[0][1]-self._virtual_points[0][1])*\
            #                                  (self._virtual_points[0][1]-self._virtual_points[1][1])/\
            #                                  (self._virtual_points[0][0]-self._virtual_points[1][0])
            # elif flag==1:
            #     self._vanishing_points[0][0]=self._virtual_points[3][0]+\
            #                                  (self._vanishing_points[0][1]-self._virtual_points[3][1])*\
            #                                  (self._virtual_points[3][1]-self._virtual_points[2][1])/\
            #                                  (self._virtual_points[3][0]-self._virtual_points[2][0])


    def get_virtual_points(self):
        self._virtual_points=self._points+self._alpha*(self._points-self._center)
        for i in range(len(self._vanishing_points)):
            self._points_center[i]+=self._vanishing_points[i]
        self._points_center/=len(self._vanishing_points)

