import copy
import functools
import itertools
import timeit

import cv2
import numpy as np
import os
from eye_utils import data_util as du

'''
we need blurred img 
GaussianBlur((3,3),1)
@glint_max_area:param pixels
'''

time_recorder = np.array([0. for i in range(5)])
times_recorder = np.array([0. for i in range(5)])
tr1 = np.array([0. for i in range(18)])
a = ['tr', 'tsr', 'tr/tsr', 'tr/tr.sum', 'ct', 'cts', 'ct/cts', 'et', 'ets', 'et/ets', 'ft', 'fts', 'ft/fts', 'fct',
     'fcts', 'fct/fcts'
    , 'pt', 'pts', 'pt/pts', 'st', 'sts', 'st/sts']

compare_vecs = np.array([[0.0, 0.28245639148673435, 0.16729039025910034, 0.2219733258380298, 0.32827989241613537],
                         [0.2824563914867344, 0.0, 0.3282798924161354, 0.22197332583802984, 0.16729039025910036],
                         [0.18197378541807696, 0.3570936418229411, 0.0, 0.15368444753933694, 0.30724812521964495],
                         [0.3055320406992954, 0.3055320406992954, 0.19446795930070465, 0.0, 0.19446795930070465],
                         [0.3570936418229411, 0.18197378541807696, 0.30724812521964495, 0.15368444753933694, 0.0]])


def get_five_points(unchosen_points):
    def cmp(a, b):
        if np.abs(a[1] - b[1]) < 5:
            if a[0] < b[0]:
                return -1
            return 1
        if a[1] < b[1]:
            return -1
        return 1

    base_point = unchosen_points[0]
    unchosen_points.sort(key=functools.cmp_to_key(cmp))
    chosen_points = []
    cross_value = 0.
    for points in itertools.combinations(unchosen_points, 5):
        index = -1
        l = []
        t = 0
        if points[0][0] < points[1][0] < points[2][0]:
            continue
        if not -5 < (points[0][1] - points[1][1]) < 5:
            continue
        values_y = np.array([points[2][1] - points[3][1], points[3][1] - points[4][1]])
        if values_y.max() > 5 or values_y.min() < -5:
            continue
        for p in points:
            if p[2] == base_point[2]:
                index = t
            dis = np.linalg.norm(base_point[0:2] - p[0:2])
            l.append(dis)
            t += 1
        if index == -1:
            continue
        l = np.array(l)
        l /= l.sum()
        if l.max() > 0.4:
            continue
        cvs = l @ compare_vecs[index]
        if cvs > cross_value:
            cross_value = cvs
            chosen_points = points
    return chosen_points


class contour_with_ellipse(object):
    def __init__(self, c, e=None):
        self.contour = c
        if e is None:
            self.ellipse = cv2.fitEllipseAMS(c)
        else:
            self.ellipse = e
        self.rth = self.ellipse[1][0] / self.ellipse[1][1]
        self.rou = 0
        self.gamma = 0
        self.theta = 0
        self.psi = -1

    # for puipil
    def get_scores(self, img):
        global tr1
        time1 = timeit.default_timer()
        self.rou = self.ellipse[1][0] / self.ellipse[1][1]
        x, y = self.ellipse[0][0], self.ellipse[0][1]
        quadrant = np.array([0.0, 0.0, 0.0, 0.0])
        c_len = len(self.contour)
        points_inline = 0.0
        points_outline = 0.0
        points = [i for i in range(c_len)]
        if c_len > 15:
            points = [i for i in range(0, c_len, c_len // 15)]
        c_len = len(points)
        for p in points:
            vec_x = self.contour[p][0][0] - x
            vec_y = self.contour[p][0][1] - y
            if vec_x > 0 and vec_y > 0:
                quadrant[0] += 1
            if vec_x > 0 and vec_y < 0:
                quadrant[1] += 1
            if vec_x < 0 and vec_y < 0:
                quadrant[2] += 1
            if vec_x < 0 and vec_y > 0:
                quadrant[3] += 1
            x = min(139, int(x))
            y = min(int(y), 139)
            _x, _y = [x, min(139, int(vec_x * 2 + x))], [int(y), min(139, int(vec_y * 2 + y))]
            inline_p, outline_p = img[_y, _x]
            points_inline += float(inline_p)
            points_outline += float(outline_p)
        quadrant /= c_len
        quadrant -= 0.25
        self.theta = (1.5 - np.abs(quadrant).sum()) / 1.5
        res = (points_outline - points_inline) / c_len
        if -70 < res <= 30:
            self.gamma = 0
        else:
            self.gamma = np.fabs(res) / 255
        self.psi = (self.rou + self.theta + self.gamma) / 3
        t2 = timeit.default_timer()
        tr1[15] += t2 - time1
        tr1[16] += 1.

    def get_rectangle(self):
        x, y = self.ellipse[0][0], self.ellipse[0][1]
        s = self.ellipse[1][0] / 2
        l = self.ellipse[1][1] / 2
        angle = self.ellipse[2] * np.pi / 180
        a_sin, a_cos = np.sin(angle), np.cos(angle)
        s_sin = abs(s * a_sin)
        s_cos = abs(s * a_cos)
        l_sin = abs(l * a_sin)
        l_cos = abs(l * a_cos)
        if (90 < self.ellipse[2] < 180) or (270 < self.ellipse[2] < 360):
            return np.array([[x + l_sin + s_cos, y + l_cos - s_sin], [x + l_sin - s_cos, y + l_cos + s_sin],
                             [x - l_sin + s_cos, y - l_cos - s_sin], [x - l_sin - s_cos, y - l_cos + s_sin]])
        else:
            return np.array([[x - l_sin + s_cos, y + l_cos + s_cos], [x - l_sin - s_cos, y + l_cos - s_cos],
                             [x + l_sin - s_cos, y - l_cos - l_sin], [x + l_sin + s_cos, y - l_cos + l_sin]])

    def fusion(self, contour, dis=None):
        if dis is None:
            self.contour = np.concatenate((contour.contour, self.contour))
            self.ellipse = cv2.fitEllipseAMS(self.contour)
            self.rth = self.ellipse[1][0] / self.ellipse[1][1]
            return True
        b_dis = (self.ellipse[0][0] - contour.ellipse[0][0]) ** 2 + (self.ellipse[0][1] - contour.ellipse[0][1]) ** 2
        if b_dis < dis:
            self.contour = np.concatenate((contour.contour, self.contour))
            self.ellipse = cv2.fitEllipseAMS(self.contour)
            self.rth = self.ellipse[1][0] / self.ellipse[1][1]
            return True
        else:
            return False

    # fusion with rectangles
    def fusion_with_rectangle(self, contour, img):
        points_1 = self.get_rectangle()
        center_2 = np.array([contour.ellipse[0][0], contour.ellipse[0][1]])
        flag = False
        dis_2 = contour.ellipse[1][0] ** 2 + contour.ellipse[1][1] ** 2
        for point in points_1:
            if ((point - center_2) ** 2).sum() < dis_2:
                flag = True
                break
        if flag:
            sub = contour_with_ellipse(self.contour, self.ellipse)
            sub.fusion(contour)
            sub.get_scores(img)
            psi = sub.psi
            if self.psi == -1:
                self.get_scores(img)
            if psi < self.psi:
                return False
            self.fusion(contour)
            self.get_scores(img)
            self.fusion(contour)
        return flag

    def add_offset(self, offset):
        center = (self.ellipse[0][0] + offset[0], self.ellipse[0][1] + offset[1])
        self.ellipse = (center, self.ellipse[1], self.ellipse[2])


def img_filter(img, filters, threshold=0, degrade=0, fill_up=0):
    filter_len = len(filters[0])
    filtered = img.copy()
    width, length = img.shape
    for i in range(width):
        for j in range(length):
            t = -1
            for filter in filters:
                t += 1
                flag = 0
                for k in range(degrade, filter_len):
                    a = i + filter[k][0]
                    b = j + filter[k][1]
                    if not (a > 0 and a < 200) or not (b > 0 and b < 200):
                        break
                    flag += int(img[a][b])
                if flag == threshold:
                    if degrade != 0:
                        for k in range(0, degrade):
                            a = i + filter[k][0]
                            b = j + filter[k][1]
                            filtered[a][b] = 0
                    if fill_up != 0:
                        for k in range(filter_len - fill_up, filter_len):
                            a = i + filter[k][0]
                            b = j + filter[k][1]
                            filtered[a][b] = 255
    return filtered


def img_filters(img, filters, threshold=None, degrade=None, fill_up=None, filter_len=None):
    assert degrade is not None
    assert fill_up is not None
    assert filter_len is not None
    if threshold is None:
        threshold = []
        for i, j in zip(degrade, filter_len):
            threshold.append((i + j) * 255)
    filters_len = len(filters)
    filtered = img.copy()
    width, length = img.shape
    for i in range(width):
        for j in range(length):
            for l in range(filters_len):
                t = -1
                for filter in filters[l]:
                    t += 1
                    flag = 0
                    for k in range(degrade[l], filter_len[l]):
                        a = i + filter[k][0]
                        b = j + filter[k][1]
                        if not (a > 0 and a < 200) or not (b > 0 and b < 200):
                            break
                        flag += int(img[a][b])
                    if flag == threshold[l]:
                        if degrade != 0:
                            for k in range(0, degrade[l]):
                                a = i + filter[k][0]
                                b = j + filter[k][1]
                                filtered[a][b] = 0
                        if fill_up != 0:
                            for k in range(filter_len[l] - fill_up[l], filter_len[l]):
                                a = i + filter[k][0]
                                b = j + filter[k][1]
                                filtered[a][b] = 255
    return filtered


class PuRe_params(object):
    def __init__(self):
        self.glints_num = 3
        self.glint_min_area = 20
        self.glint_max_area = 100
        self.gd_min = 5
        self.gd_max = 10
        self.pd_min = 30
        self.pd_max = 60
        self.puipil_min_area = 720
        self.pupil_max_area = 3000
        self.find_contour_param = cv2.CHAIN_APPROX_TC89_KCOS
        self.r_th = 0.5
        self.threshold1 = 40
        self.threshold2 = 80
        self.kernel = np.zeros((11, 11), dtype=np.float32)
        self.g_threshold = 50
        self.p_binary_threshold = 35
        self.g_binary_threshold = 135
        self.pupil_d_l = 0.
        self.pupil_d_r = 0.
        for i in range(11):
            for j in range(11):
                if (np.abs(i - 5) ** 2) + (np.abs(j - 5) ** 2) <= 16:
                    self.kernel[i][j] = 1.0 / 40.0
                else:
                    self.kernel[i][j] = -1.0 / 60.0


class PuRe(object):
    def __init__(self, params=None):
        if params is not None:
            self.set_params(params)
        else:
            self.gd_min = 5
            self.gd_max = 10
            self.pd_min = 30
            self.pd_max = 60
            self.glints_num = 3
            self.glint_min_area = 10
            self.glint_max_area = 100
            self.puipil_min_area = 700
            self.pupil_max_area = 3000
            self.find_contour_param = cv2.CHAIN_APPROX_TC89_KCOS
            self.r_th = 0.5
            self.threshold1 = 40
            self.threshold2 = 80
            self.g_threshold = 50
            self.kernel = np.zeros((11, 11), dtype=np.float32)
            self.p_binary_threshold = 35
            self.g_binary_threshold = 135
            self.pupil_center = np.zeros(2, dtype=np.float32)
            self.pupil_d_l = 0.
            self.pupil_d_r = 0.
            for i in range(11):
                for j in range(11):
                    if (np.abs(i - 5) ** 2) + (np.abs(j - 5) ** 2) <= 16:
                        self.kernel[i][j] = 1.0 / 40.0
                    else:
                        self.kernel[i][j] = -1.0 / 60.0
        self.filter_1 = [[[0, 0], [2, -1], [1, -1], [1, -2]],
                         [[0, 0], [-2, 1], [-1, 1], [-1, 2]],
                         [[0, 0], [1, 2], [1, 1], [2, 1]],
                         [[0, 0], [-1, -2], [-1, -1], [-2, -1]]]
        self.filter_2 = [[[[0, 0], [-1, 1], [1, 1], [0, 1]],
                          [[0, 0], [1, 1], [1, -1], [1, 0]],
                          [[0, 0], [1, -1], [-1, -1], [0, -1]],
                          [[0, 0], [-1, -1], [-1, 1], [-1, 0]]],

                         [[[0, 0], [0, 1], [1, 2], [1, -1], [1, 0], [1, 1]],
                          [[0, 0], [-1, 0], [-2, 1], [1, 1], [0, 1], [-1, 1]],
                          [[0, 0], [1, 0], [2, -1], [-1, -1], [0, -1], [1, -1]],
                          [[0, 0], [0, -1], [-1, -2], [-1, 1], [-1, 0], [-1, -1]]]]
        self.filter_3 = [[[[0, 0], [1, 1], [2, 1], [0, -1]],
                          [[0, 0], [-1, -1], [-2, -1], [0, 1]],
                          [[0, 0], [0, 1], [1, -1], [2, -1]],
                          [[0, 0], [0, -1], [-2, 1], [-1, 1]]],
                         [[[0, 0], [1, 1], [2, 1], [3, 1], [-1, -3], [-1, -2], [-1, -1]],
                          [[0, 0], [-1, -1], [-2, -1], [-3, -1], [1, 1], [1, 2], [1, 3]],
                          [[0, 0], [-1, 1], [-1, 2], [-1, 3], [3, -1], [2, -1], [1, -1]],
                          [[0, 0], [1, -1], [1, -2], [1, -3], [-3, 1], [-2, 1], [-1, 1]]],
                         [[[0, 0], [1, 1], [2, 2], [2, -2], [1, -1]],
                          [[0, 0], [-1, 1], [-2, 2], [2, 2], [1, 1]],
                          [[0, 0], [1, -1], [2, -2], [-2, -2], [-1, -1]],
                          [[0, 0], [-1, -1], [-1, -1], [-2, 2], [-1, 1]]],
                         [[[0, 0], [1, 1], [2, 2], [2, -3], [1, -2], [0, -1]],
                          [[0, 0], [-1, 1], [-2, 2], [3, 2], [2, 1], [1, 0]],
                          [[0, 0], [1, -1], [2, -2], [-3, -2], [-2, -1], [-1, 0]],
                          [[0, 0], [-1, -1], [-2, -2], [-2, 3], [-1, 2], [0, 1]]]]

    def set_params(self, params):
        self.gd_min = params.gd_min
        self.gd_max = params.gd_max
        self.pd_min = params.pd_min
        self.pd_max = params.pd_max
        self.glints_num = params.glints_num
        self.glint_min_area = params.glint_min_area
        self.glint_max_area = params.glint_max_area
        self.puipil_min_area = params.puipil_min_area
        self.pupil_max_area = params.pupil_max_area
        self.find_contour_param = params.find_contour_param
        self.r_th = params.r_th
        self.threshold1 = params.threshold1
        self.threshold2 = params.threshold2
        self.kernel = params.kernel
        self.g_threshold = params.g_threshold
        self.p_binary_threshold = params.p_binary_threshold
        self.g_binary_threshold = params.g_binary_threshold
        self.pupil_center=np.zeros(2,dtype=np.float32)

    def filter_thin(self, img):
        filter_img = img_filter(img, self.filter_1, 255 * 3, degrade=1)
        return filter_img

    def filter_straight(self, img):
        filter_img = img_filters(img, self.filter_2, degrade=[1, 1], fill_up=[2, 2], filter_len=[4, 4])
        return filter_img

    def filter_break_up(self, img):
        filter_img = img_filters(img, self.filter_3, degrade=[1, 1, 1, 1], fill_up=[0, 0, 0, 0],
                                 filter_len=[4, 7, 5, 6])
        return filter_img

    # binary img
    def find_contours(self, img, d_min, d_max, rec_filter=False):
        g_contours = []
        global tr1
        time1 = timeit.default_timer()
        canny_img = cv2.Canny(img, self.threshold1, self.threshold2)
        time2 = timeit.default_timer()
        tr1[0] += time2 - time1
        tr1[1] += 1.0
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_LIST, self.find_contour_param)
        time3 = timeit.default_timer()
        tr1[3] += time3 - time2
        tr1[4] += 1.
        c_len = len(contours)
        for i in range(c_len):
            if len(contours[i]) < 5:
                continue
            time4 = timeit.default_timer()
            ellipse = cv2.fitEllipseAMS(contours[i])
            time5 = timeit.default_timer()
            tr1[6] += time5 - time4
            tr1[7] += 1.
            x = ellipse[0][1]
            y = ellipse[0][0]
            horizon = np.fabs(ellipse[1][1] * np.cos(ellipse[2] * np.pi / 180))
            vertical = np.fabs(ellipse[1][1] * np.sin(ellipse[2] * np.pi / 180))
            nums = np.array([x, y, y + horizon, y - horizon, x + vertical, x - vertical])
            if np.max(nums) >= 140:
                continue
            if np.min(nums) < 0:
                continue
            if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
                continue
            if img[int(x)][int(y)] < 20:
                if ellipse[1][0] > 10:
                    if x+10<140 and img[int(x)+10][int(y)] < 20:
                        continue
                    if x-10>0 and img[int(x)-10][int(y)] < 20:
                        continue
                else:
                    continue

            # s = np.fabs(cv2.contourArea(contours[i], True))
            # s_e = np.pi * ellipse[1][0] * ellipse[1][1]
            # if s < s_e * 0.25 * 0.5:
            #     continue
            rth = ellipse[1][0] / ellipse[1][1]
            if rth < self.r_th:
                continue
            if not d_min <= (ellipse[1][0]) <= d_max:
                continue
            if not d_min <= (ellipse[1][1]) <= d_max:
                continue
            g_contours.append(contour_with_ellipse(c=contours[i], e=ellipse))

        return self.fusion_contours(g_contours)

    ## pupil->list [x,y,d**2] img:origin img

    def fusion_contours(self, contours):
        global tr1
        time1 = timeit.default_timer()
        c_len = len(contours)
        flags = [True for i in range(c_len)]
        g_contours = []
        for i in range(c_len):
            if flags[i] == False:
                continue
            for j in range(i + 1, c_len):
                res = contours[i].fusion(contours[j], dis=5)
                if res == True:
                    flags[j] = False
            g_contours.append(contours[i])
        time2 = timeit.default_timer()
        tr1[6] += time2 - time1
        tr1[7] += 1.
        return g_contours

    def get_glints(self, img, contours, pupil):
        glint_contours = []
        flags = []
        for contour in contours:
            dis = ((pupil[0] - contour.ellipse[0][0]) ** 2) + ((pupil[1] - contour.ellipse[0][1]) ** 2)
            if dis > pupil[2]:
                continue
            flags.append([contour, dis])
        flags.sort(key=lambda x: x[1])
        points = []
        if len(flags) < 5:
            return flags
        t = 0
        for flag in flags:
            points.append(np.array([flag[0].ellipse[0][0], flag[0].ellipse[0][1], t], dtype=np.float32))
            t += 1
        contour_flags = get_five_points(points)
        for cf in contour_flags:
            glint_contours.append(flags[int(cf[2])][0])
        return glint_contours[0:self.glints_num]

    # origin img
    def get_pupils(self, contours, img):
        c_len = len(contours)
        index = 0
        psi = -1
        if c_len == 0:
            return False
        for i in range(c_len):
            contours[i].get_scores(img)
            if contours[i].psi > psi:
                psi = contours[i].psi
                index = i
        return contours[index]

    # gray img
    def detect(self, img):
        global times_recorder, time_recorder
        time1 = timeit.default_timer()
        pupil_img = cv2.threshold(img, self.p_binary_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        res1 = cv2.connectedComponentsWithStats(pupil_img, connectivity=8)
        time2 = timeit.default_timer()
        times_recorder[0] += 1.0
        time_recorder[0] += time2 - time1
        pupils = []
        for contours, centroid in zip(res1[2], res1[3]):
            if not self.pd_min < contours[2] < self.pd_max:
                continue
            if not self.pd_min < contours[3] < self.pd_max:
                continue
            min_s = contours[2] * contours[3]
            if contours[4] < min_s * 0.6:
                continue
            x = contours[0] + contours[2] / 2
            y = contours[1] + contours[3] / 2
            if 0 <= centroid[0] <= 70 or 1850 <= centroid[0] <= 1920:
                continue
            if 0 <= centroid[1] <= 70 or 1010 <= centroid[1] <= 1080:
                continue
            dis = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
            if dis > 20:
                continue
            pupils.append(centroid)
        if len(pupils) <= 1:
            return False
        time3 = timeit.default_timer()
        times_recorder[1] += 1.0
        time_recorder[1] += time3 - time2
        # cv2.setNumThreads(2)
        self.pupil_center = np.array([pupils[0][0] + pupils[1][0], pupils[0][1] + pupils[1][1]], dtype=np.float32)
        self.pupil_center /= 2.
        img_1 = img[int(pupils[0][1]) - 70:int(pupils[0][1]) + 70, int(pupils[0][0]) - 70:int(pupils[0][0]) + 70]
        img_2 = img[int(pupils[1][1]) - 70:int(pupils[1][1]) + 70, int(pupils[1][0]) - 70:int(pupils[1][0]) + 70]
        img_1_origin = (int(pupils[0][0]) - 70, int(pupils[0][1]) - 70)
        img_2_origin = (int(pupils[1][0]) - 70, int(pupils[1][1]) - 70)
        img_1 = cv2.GaussianBlur(img_1, (3, 3), 3)
        img_2 = cv2.GaussianBlur(img_2, (3, 3), 3)
        glint_img_1 = cv2.threshold(img_1, self.g_binary_threshold, 255, cv2.THRESH_BINARY)[1]
        glint_img_2 = cv2.threshold(img_2, self.g_binary_threshold, 255, cv2.THRESH_BINARY)[1]
        best_contours1 = []
        best_contours2 = []
        benchmark_1 = 0
        benchmark_2 = 0
        time4 = timeit.default_timer()
        time_recorder[2] += time4 - time3
        times_recorder[2] += 1.0
        # cv2.setNumThreads(2)
        for i in range(40, 56, 5):
            pupil_img_1 = cv2.threshold(img_1, i, 255, cv2.THRESH_BINARY_INV)[1]
            pupil_img_2 = cv2.threshold(img_2, i, 255, cv2.THRESH_BINARY_INV)[1]
            tf1 = timeit.default_timer()
            pupil_contours_1 = self.find_contours(pupil_img_1, self.pd_min, self.pd_max)
            pupil_contours_2 = self.find_contours(pupil_img_2, self.pd_min, self.pd_max)
            tf2 = timeit.default_timer()
            tr1[9] += tf2 - tf1
            tr1[10] += 1.
            p_contours_1 = self.get_pupils(pupil_contours_1, img_1)
            p_contours_2 = self.get_pupils(pupil_contours_2, img_2)
            tr3 = timeit.default_timer()
            tr1[12] += tr3 - tf2
            tr1[13] += 1.
            if not isinstance(p_contours_1, bool):
                b1 = p_contours_1.ellipse[1][0] / p_contours_1.ellipse[1][1]
                if b1 > benchmark_1:
                    benchmark_1 = b1
                    best_contours1 = p_contours_1
            if not isinstance(p_contours_2, bool):
                b2 = p_contours_2.ellipse[1][0] / p_contours_2.ellipse[1][1]
                if b2 > benchmark_2:
                    benchmark_2 = b2
                    best_contours2 = p_contours_2
            if benchmark_1 > 0.95 and benchmark_2 > 0.95:
                break
        # cv2.parallel.setParallelForBackend()
        p_contours_1 = best_contours1
        p_contours_2 = best_contours2
        time5 = timeit.default_timer()
        times_recorder[3] += 1.0
        time_recorder[3] += time5 - time4
        if isinstance(p_contours_1, list) or isinstance(p_contours_2, list):
            return False
        self.pupil_d_l = (p_contours_1.ellipse[1][0] + p_contours_1.ellipse[1][1]) / 2.
        self.pupil_d_r = (p_contours_2.ellipse[1][0] + p_contours_2.ellipse[1][1]) / 2.
        glint_contours_1 = self.find_contours(glint_img_1, self.gd_min, self.gd_max)
        glint_contours_2 = self.find_contours(glint_img_2, self.gd_min, self.gd_max)
        if isinstance(p_contours_2, bool) or isinstance(p_contours_1, bool):
            return False
        g_contours_1 = []
        g_contours_2 = []
        if isinstance(p_contours_1, contour_with_ellipse):
            pupil = [p_contours_1.ellipse[0][0], p_contours_1.ellipse[0][1],
                     (((p_contours_1.ellipse[1][0] + p_contours_1.ellipse[1][1]) / 2) ** 2) * 4]
            g_contours_1 = self.get_glints(img_1, glint_contours_1, pupil)
        if isinstance(p_contours_2, contour_with_ellipse):
            pupil = [p_contours_2.ellipse[0][0], p_contours_2.ellipse[0][1],
                     (((p_contours_2.ellipse[1][0] + p_contours_2.ellipse[1][1]) / 2) ** 2) * 4]
            g_contours_2.extend(self.get_glints(img_2, glint_contours_2, pupil))
        if len(g_contours_1) < 5 or len(g_contours_2) < 5:
            return False
        p_contours_1.add_offset(img_1_origin)
        p_contours_2.add_offset(img_2_origin)
        for contour in g_contours_1:
            contour.add_offset(img_1_origin)
        for contour in g_contours_2:
            contour.add_offset(img_2_origin)
        time6 = timeit.default_timer()
        times_recorder[4] += 1.0
        time_recorder[4] += time6 - time5
        return (g_contours_1, p_contours_1), (g_contours_2, p_contours_2)


def draw_ellipse(drawed_img, c):
    for contour in c:
        center = (np.ceil(contour.ellipse[0][0]), np.ceil(contour.ellipse[0][1]))
        axes = (np.ceil(contour.ellipse[1][0]), np.ceil(contour.ellipse[1][1]))
        drawed_img = cv2.ellipse(drawed_img, [center, axes, contour.ellipse[2]], color=(0, 255, 0))
    return drawed_img


def get_time():
    res1 = [time_recorder.tolist(), times_recorder.tolist(), (time_recorder / times_recorder).tolist(),
            (time_recorder / time_recorder.sum()).tolist()]
    for i in range(0, 16, 3):
        tr1[i + 2] = tr1[i] / tr1[i + 1]
    res1.extend(tr1.tolist())
    res = []
    for i, j in zip(a, res1):
        res.append((i, j))
    return res
