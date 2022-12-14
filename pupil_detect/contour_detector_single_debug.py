import copy
import functools

import cv2
import numpy as np
import os
from eye_utils import data_util as du
import itertools

'''
we need blurred img 
GaussianBlur((3,3),1)
@glint_max_area:param pixels
'''
compare_vecs = np.array([[0.0, 0.28245639148673435, 0.16729039025910034, 0.2219733258380298, 0.32827989241613537],
                         [0.2824563914867344, 0.0, 0.3282798924161354, 0.22197332583802984, 0.16729039025910036],
                         [0.18197378541807696, 0.3570936418229411, 0.0, 0.15368444753933694, 0.30724812521964495],
                         [0.3055320406992954, 0.3055320406992954, 0.19446795930070465, 0.0, 0.19446795930070465],
                         [0.3570936418229411, 0.18197378541807696, 0.30724812521964495, 0.15368444753933694, 0.0]])


def get_five_points(unchosen_points):
    print(unchosen_points)
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

    print(f'this is poitns len {len(unchosen_points)}')
    for points in itertools.combinations(unchosen_points, 5):
        index = -1
        l = []
        t = 0
        if points[0][0] <points[1][0] <points[2][0]:
            continue
        if not -5<(points[0][1]-points[1][1])<5:
            continue
        values_y=np.array([points[2][1]-points[3][1],points[3][1]-points[4][1]])
        print(f'this is values_y {values_y}')
        if values_y.max()>5 or values_y.min()<-5:
            continue
        for p in points:
            if p[2] == base_point[2]:
                index = t
            dis = np.linalg.norm(base_point[0:2] - p[0:2])
            l.append(dis)
            t += 1
        if index==-1:
            continue
        l = np.array(l)
        l /= l.sum()
        print(f'this is l {l}')
        if l.max()>0.4:
            continue
        cvs = l @ compare_vecs[index]
        if cvs > cross_value:
            cross_value = cvs
            chosen_points = points
    print(f'this is cross_value{cross_value}')
    # if cross_value < 0.2:
    #     return []
    return chosen_points
    vectors = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
    u_len = len(unchosen_points)
    flags = [False for i in range(u_len)]
    flags[0] = True
    threshold1 = np.cos(np.pi / 8)
    for p in chosen_points:
        for i in range(u_len):
            if flags[i]:
                continue
            vec = unchosen_points[i] - p
            flag = False
            for vector in vectors:
                angle = (vec @ vector) / np.linalg.norm(vec)
                if angle > threshold1:
                    flag = True
                    break
            if not flag:
                continue
            flags[i] = True
            flag = 0
            dirs = [False, False, False, False]
            for j in range(u_len):
                vec = unchosen_points[i] - unchosen_points[j]
                if np.linalg.norm(vec) == 0:
                    continue
                for k in range(4):
                    if dirs[k]:
                        continue
                    angle = (vec @ vectors[k]) / np.linalg.norm(vec)
                    if angle > threshold1:
                        flag += 1
                        dirs[k] = True
            if flag >= 2:
                chosen_points.append(unchosen_points[i])
            else:
                flags[i] = False
    return flags



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
        print(res)
        if -70 < res <= 30:
            self.gamma = 0
        else:
            self.gamma = np.fabs(res) / 255
        self.psi = (self.rou + self.theta + self.gamma) / 3

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
        center = (np.ceil(self.ellipse[0][0]) + offset[0], np.ceil(self.ellipse[0][1]) + offset[1])
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
                    if not (a > 0 and a < 140) or not (b > 0 and b < 140):
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
                        if not (a > 0 and a < 140) or not (b > 0 and b < 140):
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
        canny_img = cv2.Canny(img, self.threshold1, self.threshold2)
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_LIST, self.find_contour_param)
        c_len = len(contours)
        i_img = np.zeros_like(img)
        i_img = cv2.cvtColor(i_img, cv2.COLOR_GRAY2RGB)
        for i in range(c_len):
            if len(contours[i]) < 5:
                continue
            ellipse = cv2.fitEllipseAMS(contours[i])
            x = ellipse[0][1]
            y = ellipse[0][0]
            horizon = abs(ellipse[1][1] * np.cos(ellipse[2] * np.pi / 180))
            vertical = abs(ellipse[1][1] * np.sin(ellipse[2] * np.pi / 180))
            nums = np.array([x, y, y + horizon, y - horizon, x + vertical, x - vertical])
            if np.max(nums) >= 140:
                continue
            if np.min(nums) < 0:
                continue
            if np.isnan(ellipse[1][0]) or np.isnan(ellipse[1][1]):
                continue
            if img[int(x)][int(y)] < 20:
                if ellipse[1][0] > 10:
                    if x + 10 < 140 and img[int(x) + 10][int(y)] < 20:
                        continue
                    if x - 10 > 0 and img[int(x) - 10][int(y)] < 20:
                        continue
                else:
                    continue
            print(f'this is pupils {ellipse}')
            rth = ellipse[1][0] / ellipse[1][1]
            if rth < self.r_th:
                continue
            if not d_min <= (ellipse[1][0]) <= d_max:
                continue
            if not d_min <= (ellipse[1][1]) <= d_max:
                continue
            if rec_filter:
                rec = cv2.minAreaRect(contours[i])
                r_f = rec[1][0] / rec[1][1]
                if r_f > 1:
                    r_f = 1 / r_f
                if r_f < 0.5:
                    continue
            a = contour_with_ellipse(c=contours[i], e=ellipse)
            i_img = draw_ellipse(i_img, [a])
            i_img = cv2.drawContours(i_img, contours, i, (255, 255, 255))
            # du.show_ph(i_img)
            g_contours.append(contour_with_ellipse(c=contours[i], e=ellipse))

        return self.fusion_contours(g_contours)

    ## pupil->list [x,y,d**2] img:origin img

    def fusion_contours(self, contours):
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
        return g_contours

    def get_glints(self, img, contours, pupil):
        glint_contours = []
        flags = []
        for contour in contours:
            dis = ((pupil[0] - contour.ellipse[0][0]) ** 2) + ((pupil[1] - contour.ellipse[0][1]) ** 2)
            if dis > pupil[2]:
                continue
            # contour.get_scores(img)
            # print(contour.ellipse, contour.gamma, dis)
            # if contour.gamma == 0:
            #     continue
            flags.append([contour, dis])
        flags.sort(key=lambda x: x[1])
        points = []
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
        pupil_img = cv2.threshold(img, self.p_binary_threshold, 255, cv2.THRESH_BINARY_INV)[1]
        res1 = cv2.connectedComponentsWithStatsWithAlgorithm(pupil_img, connectivity=8, ltype=cv2.CV_32S,
                                                             ccltype=cv2.CCL_DEFAULT)
        du.show_ph(pupil_img)
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
            dis = ((x - centroid[0]) ** 2) + ((y - centroid[1]) ** 2)
            if dis > 20:
                continue
            pupils.append(centroid)
        if len(pupils) <= 1:
            return False
        color_i = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rec_i = cv2.rectangle(color_i, (int(pupils[0][0]) - 70, int(pupils[0][1]) - 70),
                              (int(pupils[0][0]) + 70, int(pupils[0][1]) + 70), color=(0, 255, 0), thickness=2)
        rec_i = cv2.rectangle(rec_i, (int(pupils[1][0]) - 70, int(pupils[1][1]) - 70),
                              (int(pupils[1][0]) + 70, int(pupils[1][1]) + 70), color=(0, 255, 0), thickness=2)
        cv2.imwrite('C:\\Users\\snapping\\Desktop\\rec.png', rec_i)
        img_1 = img[int(pupils[0][1]) - 70:int(pupils[0][1]) + 70, int(pupils[0][0]) - 70:int(pupils[0][0]) + 70]
        img_2 = img[int(pupils[1][1]) - 70:int(pupils[1][1]) + 70, int(pupils[1][0]) - 70:int(pupils[1][0]) + 70]
        img_1_origin = (int(pupils[0][0]) - 70, int(pupils[0][1]) - 70)
        img_2_origin = (int(pupils[1][0]) - 70, int(pupils[1][1]) - 70)
        cv2.imwrite('C:\\Users\\snapping\\Desktop\\left.png', img_1)
        cv2.imwrite('C:\\Users\\snapping\\Desktop\\right.png', img_2)
        img_1 = cv2.GaussianBlur(img_1, (3, 3), 3)
        img_2 = cv2.GaussianBlur(img_2, (3, 3), 3)
        best_contours1 = []
        best_contours2 = []
        benchmark_1 = 0
        benchmark_2 = 0
        for i in range(60, 91, 10):
            pupil_img_1 = cv2.threshold(img_1, i, 255, cv2.THRESH_BINARY_INV)[1]
            pupil_img_2 = cv2.threshold(img_2, i, 255, cv2.THRESH_BINARY_INV)[1]
            du.show_ph(pupil_img_1)
            du.show_ph(pupil_img_2)
            pupil_contours_1 = self.find_contours(pupil_img_1, self.pd_min, self.pd_max, True)
            pupil_contours_2 = self.find_contours(pupil_img_2, self.pd_min, self.pd_max, True)
            if len(pupil_contours_1)==0 or len(pupil_contours_2)==0:
                continue
            p_contours_1 = self.get_pupils(pupil_contours_1, img_1)
            p_contours_2 = self.get_pupils(pupil_contours_2, img_2)
            i1 = img_1.copy()
            i2 = img_2.copy()
            if not isinstance(p_contours_1, bool):
                i1 = draw_ellipse(i1, [p_contours_1])
                # du.show_ph(i1,name='left')
                b1 = p_contours_1.ellipse[1][0] / p_contours_1.ellipse[1][1]
                if b1 > benchmark_1:
                    benchmark_1 = b1
                    best_contours1 = p_contours_1
            if not isinstance(p_contours_2, bool):
                i2 = draw_ellipse(i2, [p_contours_2])
                # du.show_ph(i2, name='left')
                b2 = p_contours_2.ellipse[1][0] / p_contours_2.ellipse[1][1]
                if b2 > benchmark_2:
                    benchmark_2 = b2
                    best_contours2 = p_contours_2
            print(f'benchmark {benchmark_1, benchmark_2} {p_contours_1} {p_contours_2}')
            if benchmark_1 > 0.95 and benchmark_2 > 0.95:
                break

        p_contours_1 = best_contours1
        p_contours_2 = best_contours2

        glint_img_1 = cv2.threshold(img_1, self.g_binary_threshold, 255, cv2.THRESH_BINARY)[1]
        glint_img_2 = cv2.threshold(img_2, self.g_binary_threshold, 255, cv2.THRESH_BINARY)[1]
        print('begin-----------------')
        glint_contours_1 = self.find_contours(glint_img_1, self.gd_min, self.gd_max)
        print(len(glint_contours_1))
        print('end-------------------')
        print('begin2-----------------')
        glint_contours_2 = self.find_contours(glint_img_2, self.gd_min, self.gd_max)
        print(len(glint_contours_2))
        print('end2-------------------')
        du.show_ph(glint_img_1,name='left')
        du.show_ph(glint_img_2,name='right')
        print(p_contours_1)
        print(p_contours_2)
        if isinstance(p_contours_2, bool) or isinstance(p_contours_1, bool):
            return False
        g_contours_1 = []
        g_contours_2 = []
        if isinstance(p_contours_1, contour_with_ellipse):
            pupil = [p_contours_1.ellipse[0][0], p_contours_1.ellipse[0][1],
                     (((p_contours_1.ellipse[1][0] + p_contours_1.ellipse[1][1]) / 2) ** 2) * 4]
            print('second------------------------------------------------------------------------------')
            g_contours_1 = self.get_glints(img_1, glint_contours_1, pupil)
            print('second------------------------------------------------------------------------------')
        if isinstance(p_contours_2, contour_with_ellipse):
            pupil = [p_contours_2.ellipse[0][0], p_contours_2.ellipse[0][1],
                     (((p_contours_2.ellipse[1][0] + p_contours_2.ellipse[1][1]) / 2) ** 2) * 4]
            print('third------------------------------------------------------------------------------')
            g_contours_2 = (self.get_glints(img_2, glint_contours_2, pupil))
            print('third------------------------------------------------------------------------------')
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2RGB)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2RGB)
        draw_img_1 = draw_ellipse(img_1, g_contours_1)
        draw_img_1 = draw_ellipse(draw_img_1, [p_contours_1])
        draw_img_2 = draw_ellipse(img_2, g_contours_2)
        draw_img_2 = draw_ellipse(draw_img_2, [p_contours_2])
        cv2.imwrite('C:\\Users\\snapping\\Desktop\\dleft.png', draw_img_1)
        cv2.imwrite('C:\\Users\\snapping\\Desktop\\dright.png', draw_img_2)
        p_contours_1.add_offset(img_1_origin)
        p_contours_2.add_offset(img_2_origin)
        for contour in g_contours_1:
            contour.add_offset(img_1_origin)
        for contour in g_contours_2:
            contour.add_offset(img_2_origin)
        return (g_contours_1, p_contours_1), (g_contours_2, p_contours_2)


def draw_ellipse(drawed_img, c):
    for contour in c:
        center = (np.ceil(contour.ellipse[0][0]), np.ceil(contour.ellipse[0][1]))
        axes = (np.ceil(contour.ellipse[1][0]), np.ceil(contour.ellipse[1][1]))
        drawed_img = cv2.ellipse(drawed_img, [center, axes, contour.ellipse[2]], color=(0, 255, 0))
    return drawed_img
