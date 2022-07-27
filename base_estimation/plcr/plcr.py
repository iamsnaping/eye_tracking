import math

import matplotlib.pyplot as plt
from gekko import GEKKO
from base_estimation.base_estimation import base_estimation
from eye_utils import utils as utils
from scipy.optimize import fsolve, root
import numpy as np


class plcr(base_estimation):
    def __init__(self, w, h, d):
        super(plcr, self).__init__(w, h, d)
        self._up = np.zeros((3, 1), np.float64)
        self._n = np.zeros((3, 1), np.float64)
        self._ez = np.zeros((3, 1), np.float64)
        self._ey = np.zeros((3, 1), np.float64)
        self._ex = np.zeros((3, 1), np.float64)
        self._fe = np.zeros((3, 3), np.float64)
        self._rt = np.float64(1.0)
        self._s = np.float64(1.0)
        # E
        # vx vy cz v=(vx,vy,0) c=(0,0,-cv)
        self._param = np.zeros((3, 1), np.float64)
        # I and I'
        self._g0 = np.zeros((3, 1), np.float64)
        self._glints = np.zeros((3, 4), np.float64)
        self._m_points = np.zeros((3, 4), np.float64)
        self._p = np.zeros((3, 1), np.float64)
        # c and v in I'
        self._ci = np.zeros((3, 1), np.float64)
        self._vi = np.zeros((3, 1), np.float64)
        #
        self._visual = np.zeros((3, 1), np.float64)

        self._plane = np.zeros((3, 1), np.float64)

        self._gaze_estimation = np.zeros((2, 2), np.float64)

        self._pupil_center=np.zeros((3,1),np.float64)

        # corneal radius actually
        self._radius=0.0

        self._intersection_points=None
        self._lights=[]

    def set_vup(self, lights):
        mid = (lights[:,0] - lights[:,3]) + (lights[:,1] - lights[:,2])
        mid.reshape((3,1))
        self._up = (mid) / np.linalg.norm(mid)
        print(f'this is up{self._up}')

    def get_param(self):
        self._s = self._rt / np.sqrt(1 - self._param[2][0] ** 2)
        print(f'this is s {self._s}')

    def get_e_coordinate(self):
        mid = self._p-self._g0
        self._n[0][0], self._n[1][0] = mid[0][0], mid[1][0]
        self._n[2][0] = np.sqrt((self._s ** 2) * (self._param[2][0] ** 2) - (np.linalg.norm(mid) ** 2))
        self._ez = self._n / np.linalg.norm(self._n)
        mid = np.cross(self._up.reshape((3)), self._ez.reshape((3)))
        self._ex = mid / np.linalg.norm(mid)
        mid = np.cross(self._ez.reshape(3), self._ex.reshape(3))
        self._ey = mid / np.linalg.norm(mid)
        self._ey = self._ey.reshape((3, 1))
        self._ex = self._ex.reshape((3, 1))
        self._fe = np.concatenate((self._ex, self._ey, self._ez), 1)
        self._fe=self._fe.T


    def transform_e_to_i(self):
        c = np.array([0, 0, -self._param[2][0]], np.float64).reshape((3, 1))
        v = np.array([self._param[0][0], self._param[1][0], 0], np.float64).reshape((3, 1))
        # leng = np.linalg.norm(c) * math.sin(2.0 * math.pi / 180) / math.cos(2 * math.pi / 180)
        # v[1][0] = v[0][0] = leng
        # v[1][0]=-v[1][0]
        self._ci = self._s * (self._fe @ c)
        self._vi = self._s * (self._fe @ v)
        print(f'this is v {self._vi}')

    def get_plane(self):
        I = []
        mid = np.array([0, 0, 1])
        for ii in range(4):
            i=self._glints[:,ii].reshape((3,1))
            r = i - self._ci
            r /= np.linalg.norm(r)
            n_i=r - 2 *(mid@r)*mid.reshape((3,1))
            I.append(n_i/np.linalg.norm(n_i))

        w=(self._w*self._s/self._radius)**2
        d=(self._w*self._s/self._radius)**2+(self._h*self._s/self._radius)**2
        h=(self._h*self._s/self._radius)**2
        def get_func(x):
            b1, b2, b3, b4 = x[0], x[1], x[2], x[3]
            return [(((self._glints[:, 0].reshape((3, 1)) + b1 * I[0]) - (
                        self._glints[:, 1].reshape((3, 1)) + b2 * I[1])) ** 2).sum() - w,
                    (((self._glints[:, 2].reshape((3, 1)) + b3 * I[2]) - (
                                self._glints[:, 3].reshape((3, 1)) + b4 * I[3])) ** 2).sum() - w,
                    (((self._glints[:, 0].reshape((3, 1)) + b1 * I[0]) - (
                                self._glints[:, 2].reshape((3, 1)) + b3 * I[2])) ** 2).sum() - d,
                    (((self._glints[:, 1].reshape((3, 1)) + b2 * I[1]) - (
                                self._glints[:, 3].reshape((3, 1)) + b4 * I[3])) ** 2).sum() - d]
        # def get_func2(x):
        #     b1, b2, b3, b4 = x[0], x[1], x[2], x[3]
        #     return [(((self._glints[:, 0].reshape((3, 1)) + b1 * I[0]) - (
        #                 self._glints[:, 1].reshape((3, 1)) + b2 * I[1])) ** 2).sum() - self._w,
        #             (((self._glints[:, 2].reshape((3, 1)) + b3 * I[2]) - (
        #                         self._glints[:, 3].reshape((3, 1)) + b4 * I[3])) ** 2).sum() - self._w,
        #             (((self._glints[:, 0].reshape((3, 1)) + b1 * I[0]) - (
        #                         self._glints[:, 3].reshape((3, 1)) + b4 * I[3])) ** 2).sum() - self._h ,
        #             (((self._glints[:, 1].reshape((3, 1)) + b2 * I[1]) - (
        #                         self._glints[:, 2].reshape((3, 1)) + b3 * I[2])) ** 2).sum() - self._h]
        # def get_func3(x):
        #     b1, b2, b3, b4 = x[0], x[1], x[2], x[3]
        #     return [(((self._glints[:, 0].reshape((3, 1)) + b1 * I[0]) - (
        #                 self._glints[:, 3].reshape((3, 1)) + b4 * I[3])) ** 2).sum() - self._h,
        #             (((self._glints[:, 2].reshape((3, 1)) + b3 * I[2]) - (
        #                         self._glints[:, 1].reshape((3, 1)) + b2 * I[1])) ** 2).sum() - self._h,
        #             (((self._glints[:, 0].reshape((3, 1)) + b1 * I[0]) - (
        #                         self._glints[:, 2].reshape((3, 1)) + b3 * I[2])) ** 2).sum() - self._h**2-self._w,
        #             (((self._glints[:, 1].reshape((3, 1)) + b2 * I[1]) - (
        #                         self._glints[:, 3].reshape((3, 1)) + b4 * I[3])) ** 2).sum() - self._h**2-self._w]
        param=self._s*60.0/self._radius*10

        # result = fsolve(get_func, [param,param, param,param])
        result = fsolve(get_func, [10000000.0, 10000000.0, 10000000.0, 10000000.0])
        print(f'this is result {result}')
        mid = np.array([0, 0, 1], np.float64).reshape((3, 1))
        points = []
        # A x =B
        print(f'this is ci{self._ci}')
        for i in range(4):
            points.append(utils.get_points_3d(self._ci, self._glints[:,i].reshape((3,1)) + result[i] * I[i], self._glints[:,i].reshape((3,1))
                          , self._glints[:,i].reshape((3,1)) + self._s*mid/self._radius).reshape((3,1)))
            self._lights.append(self._glints[:,i].reshape((3,1))+result[i]*I[i])
        print(f' this is points_intersection{points}')
        A = np.zeros((3, 3), np.float64)
        B = np.zeros((3, 1), np.float64)
        self._intersection_points=points
        for i in range(4):
            A[0, 0] = A[0, 0] + points[i][0][0] ** 2
            A[0, 1] = A[0, 1] + points[i][0][0] * points[i][1][0]
            A[0, 2] = A[0, 2] + points[i][0][0]
            A[1, 0] = A[1, 0] + points[i][0][0] * points[i][1][0]
            A[1, 1] = A[1, 1] + points[i][1][0] ** 2
            A[1, 2] = A[1, 2] + points[i][1][0]
            A[2, 0] = A[2, 0] + points[i][0][0]
            A[2, 1] = A[2, 1] + points[i][1][0]
            A[2, 2] = 4.0
            B[0, 0] = B[0, 0] + points[i][0][0] * points[i][2][0]
            B[1, 0] = B[1, 0] + points[i][1][0] * points[i][2][0]
            B[2, 0] = B[2, 0] + points[i][2][0]
        A_inv = np.linalg.inv(A)
        self._plane =A_inv@B
        for i in range(4):
            t=self._plane[0][0]*points[i][0][0]+self._plane[1][0]*points[i][1][0]+self._plane[2][0]
            t=points[i][2][0]-t
            print(f'this is point i:{i} and bias {t}')
        print(f'this is plane {self._plane}')

    def get_visual(self):
        I = np.array([self._plane[0][0], self._plane[1][0], -1.0], np.float64)
        I=I/np.linalg.norm(I)
        print(f'this is I{I}')
        mid = self._vi.reshape(3)-self._ci.reshape(3)
        point = np.array([1, 1, self._plane.sum()], np.float64)
        c = point - self._ci.reshape(3)
        c/=np.linalg.norm(c)
        mid/=np.linalg.norm(mid)
        d = np.linalg.norm(np.cross(c, I)) / np.linalg.norm(np.cross(mid, I))
        self._visual = self._ci + d * I.reshape((3, 1))
        self._visual[2][0] = 0
        print(f'this is visual {self._visual}')

    def get_m_points(self):
        self._glints=self._pupil_center-self._glints
        self._visual = self._pupil_center - self._visual
        print(f'this is visual2 {self._visual}')
        g = self._glints.T
        print(f'this is g{g}')
        v = []
        self._visual=self._pupil_center-self._visual
        v.append(utils.get_points_3d(g[0], g[1], g[2], g[3]))
        v.append(utils.get_points_3d(g[1], g[2], g[3], g[0]))
        c = g[0] + g[1] + g[2] + g[3]
        c /= 4.0
        self._m_points[:,0] = utils.get_points_3d(g[0], g[1], c, v[1])
        self._m_points[:,1] = utils.get_points_3d(g[0], g[1], self._visual, v[1])
        self._m_points[:,2] = utils.get_points_3d(v[0], self._visual, g[1], g[2])
        self._m_points[:,3] = utils.get_points_3d(v[0], c, g[1], g[2])
        print(f"this is m_points {self._m_points}")
        # x=[v[0][0],v[1][0]]
        # y=[v[0][1],v[1][1]]
        x=[]
        y=[]
        col=[]
        for i in range(4):
            x.append(self._m_points[0,i])
            y.append(self._m_points[1,i])
            col.append(2)
        for i in range(4):
            x.append(g[i,0])
            y.append(g[i,1])
            col.append(3)
        ci=self._pupil_center-self._ci
        vi=self._pupil_center-self._vi
        x.append(ci[0][0])
        x.append(vi[0][0])
        y.append(ci[1][0])
        y.append(vi[1][0])
        col.append(4)
        col.append(5)
        for point in self._intersection_points:
            x.append(point[0][0]+self._pupil_center[0][0])
            y.append(point[1][0]+self._pupil_center[1][0])
            col.append(6)
        k=7
        # for light in self._lights:
        #     x.append(light[0][0]+self._pupil_center[0][0])
        #     y.append(light[1][0]+self._pupil_center[1][0])
        #     col.append(k)
        #     k+=1
        plt.scatter(y,x,c=col,s=5)
        plt.show()


    def gaze_estimation(self):
        tu = self.cross_ratio(self._glints, self._m_points)
        c_x, c_y = tu[0], tu[1]
        print(f'this is c_x and c_y{c_x, c_y}')
        self._gaze_estimation[0][0] = (self._w * c_x) / (1.0 + c_x)
        self._gaze_estimation[1][0] = (self._h * c_y) / (1.0 + c_y)
        return self._gaze_estimation

# calibration
# eqs = []
# for i in range(2):
#     eqs.append(Eq((((self._glints[0 + i * 2] + B[0 + i * 2] * I[0 + i * 2]) - (
#                 self._glints[1 + i * 2] + B[1 + i * 2] * I[1 + i * 2])) ** 2).sum()), self._w ** 2)
#     eqs.append(Eq((((self._glints[0 + i * 1] + B[0 + i * 1] * I[0 + i * 1]) - (
#                 self._glints[3 - i * 1] + B[3 - i * 1] * I[3 - i * 1])) ** 2).sum()), self._h ** 2)
#     eqs.append(Eq((((self._glints[0 + i * 1] + B[0 + i * 1] * I[0 + i * 1]) - (
#                 self._glints[2 + i * 1] + B[2 + i * 1] * I[2 + i * 1])) ** 2).sum()), self._d ** 2)
# result = solve(eqs, b)


