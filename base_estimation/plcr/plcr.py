from base_estimation.base_estimation import base_estimation
from eye_utils import utils as utils
from scipy.optimize import fsolve, root
import numpy as np

time_recorder=np.array([0.,0.,0.,0.])
times_recorder=np.array([0.,0.,0.,0.])


class plcr(base_estimation):
    def __init__(self, w=34.0, h=27.0):
        d = np.sqrt(w ** 2 + h ** 2)
        super(plcr, self).__init__(w, h, d)
        self._up = np.zeros((3, 1), np.float32)
        self._n = np.zeros((3, 1), np.float32)
        self._ez = np.zeros((3, 1), np.float32)
        self._ey = np.zeros((3, 1), np.float32)
        self._ex = np.zeros((3, 1), np.float32)
        self._fe = np.zeros((3, 3), np.float32)
        self._rt = np.float32(1.0)
        self._s = np.float32(1.0)
        # E
        # vx vy cz v=(vx,vy,0) c=(0,0,-cv)
        self._param = np.zeros((3, 1), np.float32)
        # I and I'
        self._g0 = np.zeros((3, 1), np.float32)
        self._glints = np.zeros((3, 4), np.float32)
        self._m_points = np.zeros((3, 4), np.float32)
        self._p = np.zeros((3, 1), np.float32)
        # c and v in I'
        self._ci = np.zeros((3, 1), np.float32)
        self._vi = np.zeros((3, 1), np.float32)
        #
        self._visual = np.zeros((3, 1), np.float32)

        self._plane = np.zeros((3, 1), np.float32)

        self._gaze_estimation = np.zeros(2, np.float32)

        self._pupil_center = np.zeros((3, 1), np.float32)

        # corneal radius actually
        self._radius = 0.0

        self._intersection_points = None
        self._lights = []
        self._bias_vec = np.zeros((2, 1)).reshape((2, 1))
        self._cali_ratio_x = 0.0
        self._cali_ratio_y = 0.0
        self._calibration_vec = []
        self._calibration_vec_des = []
        self._is_calibration = True
        self._cross_ratio_points = []

    def refresh(self):
        self._up = np.zeros((3, 1), np.float32)
        self._n = np.zeros((3, 1), np.float32)
        self._ez = np.zeros((3, 1), np.float32)
        self._ey = np.zeros((3, 1), np.float32)
        self._ex = np.zeros((3, 1), np.float32)
        self._fe = np.zeros((3, 3), np.float32)
        self._s = np.float32(1.0)
        # E
        # vx vy cz v=(vx,vy,0) c=(0,0,-cv)
        self._param = np.zeros((3, 1), np.float32)
        # I and I'
        self._g0 = np.zeros((3, 1), np.float32)
        self._glints = np.zeros((3, 4), np.float32)
        self._m_points = np.zeros((3, 4), np.float32)
        self._p = np.zeros((3, 1), np.float32)
        # c and v in I'
        self._ci = np.zeros((3, 1), np.float32)
        self._vi = np.zeros((3, 1), np.float32)
        #
        self._visual = np.zeros((3, 1), np.float32)

        self._plane = np.zeros((3, 1), np.float32)

        self._gaze_estimation = np.zeros(2, np.float32)

        self._pupil_center = np.zeros((3, 1), np.float32)
        self._intersection_points = None
        self._lights = []
        self._bias_vec = np.zeros((2, 1)).reshape((2, 1))
        self._cali_ratio_x = 0.0
        self._cali_ratio_y = 0.0
        self._cross_ratio_points = []
        self._is_calibration=True

    def set_vup(self, lights):
        mid = (lights[:, 0] - lights[:, 3]) + (lights[:, 1] - lights[:, 2])
        mid.reshape((3, 1))
        self._up = (mid) / np.linalg.norm(mid)

    # 5 1.5 /  2 0.6
    def calibration(self, x, y):
        self._cali_ratio_x = x
        self._cali_ratio_y = y

    def set_bias_vec(self, b_v):
        self._bias_vec = b_v

    def get_param(self):
        self._s = self._rt / np.sqrt(1 - self._param[2][0] ** 2)

    def get_e_coordinate(self):
        mid = self._p - self._g0
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
        self._fe = self._fe.T

    def transform_e_to_i(self):
        c = np.array([0, 0, -self._param[2][0]], np.float32).reshape((3, 1))
        v = np.array([self._param[0][0], self._param[1][0], 0], np.float32).reshape((3, 1))
        leng = np.linalg.norm(c)
        if self._cali_ratio_x != 0.0 and self._cali_ratio_y != 0.0:
            v[0][0] = self._cali_ratio_x * leng
            v[1][0] = self._cali_ratio_y * leng
        self._ci = self._s * (self._fe @ c)
        self._vi = self._s * (self._fe @ v)

    def get_plane(self):
        I = []
        mid = np.array([0, 0, -1], dtype=np.float32)
        # r 法向量
        dci = self._ci.reshape(3)
        for ii in range(4):
            i = self._glints[:, ii]
            r = i - dci
            r /= np.linalg.norm(r)
            n_i = mid - 2 * np.dot(mid, r) * r
            I.append(n_i.reshape((3, 1)) / np.linalg.norm(n_i))
        w = (self._w * self._s / self._radius) ** 2
        d = (self._w * self._s / self._radius) ** 2 + (self._h * self._s / self._radius) ** 2
        h = (self._h * self._s / self._radius) ** 2

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

        param = self._s * np.sqrt(60.0 ** 2 + 27.0 ** 2 + 17.0 ** 2) / self._radius
        # param=np.sqrt()
        param *= 1.5
        result = fsolve(get_func, [param, param, param, param])
        mid = np.array([0, 0, 1], np.float32).reshape((3, 1))
        points = []
        # A x =B
        for i in range(4):
            points.append(utils.get_points_3d(self._ci, self._glints[:, i].reshape((3, 1)) + result[i] * I[i],
                                              self._glints[:, i].reshape((3, 1))
                                              , self._glints[:, i].reshape(
                    (3, 1)) + self._s * mid / self._radius).reshape((3, 1)))
            self._lights.append(self._glints[:, i].reshape((3, 1)) + result[i] * I[i])
        A = np.zeros((3, 3), np.float32)
        B = np.zeros((3, 1), np.float32)
        self._intersection_points = points
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
        self._plane = A_inv @ B

    def get_visual(self):
        # I_cv cv 向量 I 平面法向量 p_v平面向量
        I = np.array([self._plane[0][0], self._plane[1][0], -1.0], np.float32)
        I = I / np.linalg.norm(I)
        I_cv = self._vi.reshape(3) - self._ci.reshape(3)
        point = np.array([1, 1, self._plane.sum()], np.float32)
        p_v = point - self._vi.reshape(3)
        p_v /= np.linalg.norm(p_v)
        I_cv /= np.linalg.norm(I_cv)
        d = np.linalg.norm(np.cross(p_v, I)) / np.linalg.norm(np.cross(I_cv, I))
        self._visual = self._vi + d * I_cv.reshape((3, 1))
        self._visual[2][0] = 0

    def get_m_points(self):
        self._glints = self._pupil_center - self._glints
        self._visual = self._pupil_center - self._visual
        g = self._glints.T
        v = []
        v.append(utils.get_points_3d(g[0], g[1], g[2], g[3]))
        v.append(utils.get_points_3d(g[1], g[2], g[3], g[0]))
        c = g[0] + g[1] + g[2] + g[3]
        c /= 4.0
        # print(f'{c} v0 {v[0].T} v1{v[1].T} visual {self._visual.T}')
        if v[1][1] < g[0][1]:
            self._m_points[:, 0] = utils.get_points_3d2(g[0], g[1], c, v[1])
            self._m_points[:, 1] = utils.get_points_3d2(g[0], g[1], self._visual, v[1])
            self._cross_ratio_points.append(g[0].reshape(3))
            self._cross_ratio_points.append(g[1].reshape(3))
        else:
            self._m_points[:, 0] = utils.get_points_3d2(g[2], g[3], c, v[1])
            self._m_points[:, 1] = utils.get_points_3d2(g[2], g[3], self._visual, v[1])
            self._cross_ratio_points.append(g[3].reshape(3))
            self._cross_ratio_points.append(g[2].reshape(3))
        if v[0][0] < g[1][0]:
            self._m_points[:, 2] = utils.get_points_3d2(g[1], g[2], self._visual, v[0])
            self._m_points[:, 3] = utils.get_points_3d2(g[1], g[2], c, v[0])
            self._cross_ratio_points.append(g[1].reshape(3))
            self._cross_ratio_points.append(g[2].reshape(3))
        else:
            self._m_points[:, 2] = utils.get_points_3d2(g[3], g[0], self._visual, v[0])
            self._m_points[:, 3] = utils.get_points_3d2(g[3], g[0], c, v[0])
            self._cross_ratio_points.append(g[0].reshape(3))
            self._cross_ratio_points.append(g[3].reshape(3))

    def gaze_estimation(self):
        tu = self.cross_ratio(self._cross_ratio_points, self._m_points.T)
        c_x, c_y = tu[0], tu[1]
        self._gaze_estimation[0] = self._w - self._w * c_x
        self._gaze_estimation[1] = self._h * c_y
        if self._is_calibration:
            return self._gaze_estimation
        # centers = np.array([52.78 / 2, 31.26 / 2], dtype=np.float32)
        # s_para = 52.78 / 1920
        centers = np.array([self._calibration_vec_des[0][0], self._calibration_vec_des[0][1]], dtype=np.float32)
        # centers = np.array([52.78 / 2, 31.26 / 2], dtype=np.float32)
        s_para = 52.78 / 1920
        compute_vec = np.zeros((2), dtype=np.float32)
        # y 480,960 x 660 1320
        # up
        # vec1 = (self.vecs[0] - self.vecs[2]) / (480 * s_para)
        vec1 = (self._calibration_vec[0] - self._calibration_vec[2])
        # down
        # vec2 = (self.vecs[5] - self.vecs[0]) / (480 * s_para)
        vec2 = (self._calibration_vec[5] - self._calibration_vec[0])
        # up
        # vec3 = (self.vecs[4] - self.vecs[1]) / (960 * s_para)
        # vec4 = (self.vecs[6] - self.vecs[3]) / (960 * s_para)
        # print(f'                  {self._calibration_vec}')
        vec3 = (self._calibration_vec[4] - self._calibration_vec[1])
        vec4 = (self._calibration_vec[6] - self._calibration_vec[3])
        center_ratio=norm(self._calibration_vec_des[0]-self._calibration_vec_des[2])/(norm(self._calibration_vec_des[5]-self._calibration_vec_des[0])+norm(self._calibration_vec_des[2]-self._calibration_vec_des[0]))
        if centers[0] > self._gaze_estimation[0]:
            left_vec=self._calibration_vec_des[1]*(1-center_ratio) + self._calibration_vec_des[4]* center_ratio
            ratio = norm(self._gaze_estimation - self._calibration_vec_des[1]) / (
                        norm(self._calibration_vec_des[2] - self._gaze_estimation) + norm(self._gaze_estimation - self._calibration_vec_des[1]))
            mid1 = ratio * self._calibration_vec_des[2] + (1 - ratio) * self._calibration_vec_des[1]
            ratio = norm(self._gaze_estimation - left_vec) / (
                        norm(self._gaze_estimation - left_vec) + norm(self._gaze_estimation - centers))
            mid2 = ratio * self._calibration_vec_des[0] + (1 - ratio) * left_vec
            ratio = norm(self._gaze_estimation - self._calibration_vec_des[4]) / (
                        norm(self._gaze_estimation - self._calibration_vec_des[4]) + norm(self._gaze_estimation - self._calibration_vec_des[5]))
            mid3 = ratio * self._calibration_vec_des[5] + (1 - ratio) * self._calibration_vec_des[4]
            scal_1 = norm(self._gaze_estimation - mid1) / norm(mid3 - mid1)
        else:
            right_vec=self._calibration_vec_des[3]*(1-center_ratio) + self._calibration_vec_des[6] * center_ratio
            ratio = norm(self._gaze_estimation - self._calibration_vec_des[2]) / (
                        norm(self._calibration_vec_des[2] - self._gaze_estimation) + norm(self._gaze_estimation - self._calibration_vec_des[3]))
            mid1 = ratio * self._calibration_vec_des[3] + (1 - ratio) * self._calibration_vec_des[2]
            ratio = norm(self._gaze_estimation - centers) / (
                        norm(self._gaze_estimation - right_vec) + norm(self._gaze_estimation - centers))
            mid2 = ratio * right_vec + (1 - ratio) * centers
            ratio = norm(self._gaze_estimation - self._calibration_vec_des[5]) / (
                        norm(self._gaze_estimation - self._calibration_vec_des[5]) + norm(self._gaze_estimation - self._calibration_vec_des[6]))
            mid3 = ratio * self._calibration_vec_des[6] + (1 - ratio) * self._calibration_vec_des[5]
            scal_1 = np.linalg.norm(self._gaze_estimation - mid1) / np.linalg.norm(mid3 - mid1)
        if self._gaze_estimation[1] < mid2[1]:
            scal_2 = norm(self._gaze_estimation - mid1) / norm(mid2 - mid1)
            vec5 = scal_2 * vec1 + self._calibration_vec[2]
            vec5_des = self._calibration_vec_des[2] + (self._calibration_vec_des[0] - self._calibration_vec_des[2]) * scal_2
        else:
            scal_2 = norm(self._gaze_estimation - mid2) / norm(mid3 - mid2)
            vec5 = scal_2 * vec2 + self._calibration_vec[0]
            vec5_des = self._calibration_vec_des[0] + (self._calibration_vec_des[5] - self._calibration_vec_des[0]) * scal_2
        if self._gaze_estimation[0] < centers[0]:
            vec6 = vec3 * scal_1 + self._calibration_vec[1]
            vec6_des = (self._calibration_vec_des[4] - self._calibration_vec_des[1]) * scal_1
            # compute_vec = (vec5 - vec6) / (660 * s_para) * gaze_estimation[0] + vec6
            compute_vec = (vec5 - vec6) / norm(vec6_des[0] - vec5_des[0]) * self._gaze_estimation[0] + vec6
        else:
            vec6 = vec4 * scal_1 + self._calibration_vec[3]
            vec6_des = (self._calibration_vec_des[6] - self._calibration_vec_des[3]) * scal_1
            # compute_vec = (vec6 - vec5) / (660 * s_para) * (gaze_estimation[0] - centers[0]) + vec5
            compute_vec = (vec6 - vec5) / norm(vec6_des[0] - vec5_des[0]) * (
                        self._gaze_estimation[0] - centers[0]) + vec5
        # print(f'sca {scal_1, scal_2,mid1,mid2,mid3,self._gaze_estimation}')
        return self._gaze_estimation - compute_vec

    def set_calibration(self, vecs, des):
        self._calibration_vec = vecs
        self._calibration_vec_des = des


def norm(a):
    return np.linalg.norm(a)