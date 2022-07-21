from base_estimation.base_estimation import base_estimation
from eye_utils import utils as utils
from sympy import symbols, Eq, solve, nsolve
from scipy.optimize import fsolve, root
import numpy as np


class plcr(base_estimation):
    def __init__(self, w, h, d):
        super(plcr, self).__init__(w, h, d)
        self._up = np.zeros((3, 1), np.float64)
        # self._cos=np.zeros((1,3),np.float64)
        # self._sin=np.zeros((1,3),np.float64)
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

    def set_vup(self, lights):
        mid = (lights[:,0] - lights[:,3]) + (lights[:,1] - lights[:,2])
        mid.reshape((3,1))
        print(mid)
        self._up = (mid) / np.linalg.norm(mid)

    def get_param(self):
        self._s = self._rt / np.sqrt(1 - self._param[2][0] ** 2)

    def get_e_coordinate(self):
        mid = self._p - self._g0
        self._n[0][0], self._n[1][0] = mid[0][0], mid[1][0]
        # print(mid)
        # print(self._param[2][0])
        # print((self._s ** 2) * (self._param[2][0] ** 2) - (np.linalg.norm(mid) ** 2))
        self._n[2][0] = np.sqrt((self._s ** 2) * (self._param[2][0] ** 2) - (np.linalg.norm(mid) ** 2))
        self._ez = self._n / np.linalg.norm(self._n)
        mid = np.multiply(self._up.reshape((1, 3)), self._ez.reshape((1, 3)))
        self._ex = mid / np.linalg.norm(mid)
        mid = np.multiply(self._ez.reshape(1, 3), self._ex)
        self._ey = mid / np.linalg.norm(mid)
        self._ey = self._ey.reshape((3, 1))
        self._ex = self._ex.reshape((3, 1))
        self._fe = np.concatenate((self._ex, self._ey, self._ez), 1)
        c = np.array([0, 0, -self._param[2][0]], np.float64).reshape((3, 1))
        v = np.array([self._param[0][0], self._param[1][0], 0], np.float64).reshape((3, 1))
        self._ci = self._s * np.dot(self._fe, c)
        self._vi = self._s * np.dot(self._fe, v)
        self._vi -= self._ci
        self._vi /= np.linalg.norm(self._vi)

    def calibration(self, param):
        self._param = param

    def get_plane(self):
        I = []
        mid = np.array([0, 0, 1])
        for ii in range(4):
            i=self._glints[:,ii].reshape((3,1))
            r = i - self._ci
            r /= np.linalg.norm(r)
            I.append(r - 2 * np.dot(mid,r) * mid.reshape((3,1)))

        def get_func(x):
            b1, b2, b3, b4 = x[0], x[1], x[2], x[3]
            return [(((self._glints[:,0].reshape((3,1)) + b1 * I[0]) - (self._glints[:,1].reshape((3,1))  + b2 * I[1])) ** 2).sum() - self._w ** 2,
             (((self._glints[:,2].reshape((3,1))  + b3 * I[2]) - (self._glints[:,3].reshape((3,1))  + b4 * I[3])) ** 2).sum() - self._w ** 2,
             (((self._glints[:,0].reshape((3,1))  + b1 * I[0]) - (self._glints[:,3].reshape((3,1))  + b4 * I[3])) ** 2).sum() - self._h ** 2,
             (((self._glints[:,1].reshape((3,1))  + b2 * I[1]) - (self._glints[:,2].reshape((3,1))  + b3 * I[2])) ** 2).sum() - self._h ** 2]

        result = fsolve(get_func, [1.0, 1.0, 1.0, 1.0])
        l = []
        r = []
        mid = np.array([0, 0, 1], np.float64).reshape((3, 1))
        points = []
        # A x =B
        for i in range(4):
            points.append(utils.get_points_3d(self._ci, self._glints[:,i].reshape((3,1)) + result[i] * I[i], self._glints[:,i].reshape((3,1))
                          , self._glints[:,i].reshape((3,1)) + mid).reshape((3,1)))
        A = np.zeros((3, 3), np.float64)
        B = np.zeros((3, 1), np.float64)
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
        self._plane = np.dot(A_inv, B)

    def get_visual(self):
        I = np.array([self._plane[0][0], self._plane[1][0], -1.0], np.float64)
        mid = self._vi.reshape(3)
        point = np.array([self._plane[0][0], self._plane[1][0], self._plane.sum()], np.float64)
        c = point - self._ci.reshape(3)
        d = np.linalg.norm(np.multiply(c, I)) / np.linalg.norm(np.multiply(self._vi.reshape(3), I))
        self._visual = self._ci + d * I.reshape((3, 1))
        self._visual[2][0] = 0

    def get_m_points(self):
        g = self._glints.reshape((4,3))
        v = []
        v.append(utils.get_points_3d(g[0], g[1], g[2], g[3]))
        v.append(utils.get_points_3d(g[1], g[2], g[3], g[0]))
        c = g[0] + g[1] + g[2] + g[3]
        c /= 4
        self._m_points[:,0] = utils.get_points_3d(v[0], g[0], c, v[1])
        self._m_points[:,1] = utils.get_points_3d(v[0], g[0], self._visual, v[1])
        self._m_points[:,2] = utils.get_points_3d(v[0], self._visual, g[1], v[1])
        self._m_points[:,3] = utils.get_points_3d(v[0], c, g[1], v[1])

    def gaze_estimation(self):
        tu = self.cross_ratio(self._glints, self._m_points)
        c_x, c_y = tu[0], tu[1]
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


