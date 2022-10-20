import copy
import math

import cv2
import numpy as np
import os
from eye_utils import data_util as du

'''
we need blurred img 
GaussianBlur((3,3),1)
@glint_max_area:param pixels
'''


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
        for points in self.contour:
            vec_x = points[0][0] - x
            vec_y = points[0][1] - y
            if vec_x > 0 and vec_y > 0:
                quadrant[0] += 1
            if vec_x > 0 and vec_y < 0:
                quadrant[1] += 1
            if vec_x < 0 and vec_y < 0:
                quadrant[2] += 1
            if vec_x < 0 and vec_y > 0:
                quadrant[3] += 1
            # front inside
            _x, _y = [int(vec_x / 2 + x), int(vec_x * 1.5 + x)], [int(vec_y / 2 + y), int(vec_y * 1.5 + y)]
            inline_p, outline_p = img[_x, _y]
            points_inline += float(inline_p)
            points_outline += float(outline_p)

        quadrant /= c_len
        quadrant -= 0.25
        self.theta = (1.5 - np.abs(quadrant).sum()) / 1.5
        res = (points_outline - points_inline) / c_len
        if res <= 30:
            self.gamma = 0
        else:
            self.gamma = res / 255
        self.psi = (self.rou + self.theta + self.gamma) / 3

    def get_rectangle(self):
        x, y = self.ellipse[0][0], self.ellipse[0][1]
        s = self.ellipse[1][0]/2
        l = self.ellipse[1][1]/2
        angle = self.ellipse[2] * math.pi / 180
        a_sin, a_cos = math.sin(angle), math.cos(angle)
        s_sin = abs(s * a_sin)
        s_cos = abs(s * a_cos)
        l_sin = abs(l * a_sin)
        l_cos = abs(l * a_cos)
        if (90 < self.ellipse[2] < 180) or (270 < self.ellipse[2] < 360):
            return np.array([[x+l_sin+s_cos,y+l_cos-s_sin],[x+l_sin-s_cos,y+l_cos+s_sin],[x-l_sin+s_cos,y-l_cos-s_sin],[x-l_sin-s_cos,y-l_cos+s_sin]])
        else:
            return np.array([[x-l_sin+s_cos,y+l_cos+s_cos],[x-l_sin-s_cos,y+l_cos-s_cos],[x+l_sin-s_cos,y-l_cos-l_sin],[x+l_sin+s_cos,y-l_cos+l_sin]])

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


def img_filter(img, filters, threshold=0, degrade=0, fill_up=0):
    filter_len = len(filters[0])
    filtered = img.copy()
    for i in range(200):
        for j in range(200):
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
    for i in range(200):
        for j in range(200):
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
        self.kernel = np.zeros((11, 11), dtype=np.float64)
        self.g_threshold = 50
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
            self.kernel = np.zeros((11, 11), dtype=np.float64)
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

    def find_contours(self, img):
        glint_contours = []
        pupil_contours = []
        img = cv2.Canny(img, self.threshold1, self.threshold2)
        du.show_ph(img,'canny_img')
        contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, self.find_contour_param)
        maxlen = math.ceil(self.pd_max * math.pi)
        c_len = len(contours)
        flags = [0 for i in range(c_len)]
        for i in range(c_len):
            if len(contours[i]) < 5:
                continue
            if len(contours[i]) > maxlen:
                continue
            ellipse = cv2.fitEllipseAMS(contours[i])
            # tes_img = np.zeros_like(img)
            # tes_img = cv2.drawContours(tes_img, contours[i], -1, (255, 255, 255))
            # tes_img=cv2.cvtColor(tes_img,cv2.COLOR_GRAY2RGB)
            # rth=ellipse[1][0]/ellipse[1][1]
            # print(f'rth: {rth}')
            # center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
            # axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
            # tes_img = cv2.ellipse(tes_img, [center, axes, ellipse[2]], color=(0, 255, 0))
            # du.show_ph(tes_img, 'tes')
            x = ellipse[0][1]
            y = ellipse[0][0]
            horizon = abs(ellipse[1][1] * math.cos(ellipse[2] * math.pi / 180))
            vertical = abs(ellipse[1][1] * math.sin(ellipse[2] * math.pi / 180))
            nums = np.array([x, y, y + horizon, y - horizon, x + vertical, x - vertical])
            if not (0 <= nums.all() < 200):
                continue
            if math.isnan(ellipse[1][0]) or math.isnan(ellipse[1][1]):
                continue
            rth = ellipse[1][0] / ellipse[1][1]
            if rth < self.r_th:
                # print('1')
                continue
            if ellipse[1][1] < self.gd_min or ellipse[1][0] > self.pd_max:
                # print('2')
                continue
            if ellipse[1][0] > self.gd_max and ellipse[1][1] < self.pd_min:
                # print('3')
                continue
            if ellipse[1][0] < self.gd_max:
                flags[i] = 2
            if ellipse[1][1] > self.pd_min:
                flags[i] = 1
        for i in range(c_len):
            if flags[i] == 0:
                continue
            son = hierarchy[0][i][2]
            parent = hierarchy[0][i][3]
            if son!=-1 and flags[i] == flags[son]:
                contours[i] = np.concatenate((contours[i], contours[son]))
            if parent!=-1 and flags[i] == flags[parent]:
                continue
            if flags[i] == 1:
                pupil_contours.append(contour_with_ellipse(contours[i]))
            else:
                glint_contours.append(contour_with_ellipse(contours[i]))
        # print(len(pupil_contours))
        # tes_img=np.zeros_like(img)
        # tes_img=cv2.drawContours(tes_img, [pupil_contours[0].contour,pupil_contours[1].contour], -1, (255, 255, 255))
        # du.show_ph(tes_img,'tes,res')
        return glint_contours, pupil_contours

    ## pupil->list [x,y,d**2]
    def get_glints(self, img, contours, pupil):
        glint_contours = []
        c_len = len(contours)
        flags = [False for i in range(c_len)]
        for i in range(c_len):
            if flags[i]:
                continue
            for j in range(i + 1, c_len):
                contour = contours[i].fusion(contours[j], 5)
                if contour == True:
                    flags[j] = True
            glint_contours.append(contours[i])
        r_glints = []
        for contour in glint_contours:
            dis = ((pupil[0] - contour.ellipse[0][0]) ** 2) + ((pupil[1] - contour.ellipse[0][1]) ** 2)
            if dis > pupil[2]:
                continue
            x, y = int(contour.ellipse[0][1]), int(contour.ellipse[0][0])
            sub = img[(x - 5):(x + 6), (y - 5):(y + 6)]
            value = (sub * self.kernel).sum()
            if value < self.g_threshold:
                continue
            r_glints.append(contour)
        return r_glints[0:self.glints_num]

    def get_pupils(self, contours, img):
        c_len = len(contours)
        index = 0
        psi = -1
        # print(c_len)
        # tes_img=np.zeros_like(img)
        # tes_img=cv2.drawContours(tes_img,[contours[0].contour,contours[1].contour],-1,(255, 255, 255))
        # tes_img=cv2.cvtColor(tes_img,cv2.COLOR_GRAY2RGB)
        # tes_img=draw_ellipse(tes_img,contours)
        # du.show_ph(tes_img)
        for i in range(c_len):
            contours[i].get_scores(img)
            for j in range(i + 1, c_len):
                contours[i].fusion_with_rectangle(contours[j], img)
                # print(f'fusion {res}')
            if contours[i].psi > psi:
                psi = contours[i].psi
                index = i
        return contours[index]

    def detect(self, img):
        img = du.get_gray_pic(img)
        img = cv2.GaussianBlur(img, (3, 3), 1)
        # binary_img=np.zeros_like(img,dtype=np.uint8)
        # for i in range(200):
        #     for j in range(200):
        #         if img[i][j]<70:
        #             binary_img[i][j]=0
        #         elif img[i][j]>=180:
        #             binary_img[i][j]=np.uint8(100)
        #         else:
        #             binary_img[i][j]=np.uint8(255)
                # binary_img[i][j] = 0 if (img[i][j] < 70 or img[i][j] >= 180) else np.uint8(255)
        # binary_img = self.filter_straight(img)
        img=self.filter_straight(img)
        glint_contours, pupil_contours = self.find_contours(img)
        print(glint_contours)
        print(pupil_contours)
        if len(pupil_contours)>=1:
            p_contours = self.get_pupils(pupil_contours, img)
            pupil = [p_contours.ellipse[0][0], p_contours.ellipse[0][1],
                     ((p_contours.ellipse[1][0] + p_contours.ellipse[1][1]) / 2) ** 2]
            g_contours = self.get_glints(img, glint_contours, pupil)
        return g_contours, p_contours


def draw_ellipse(drawed_img, c,offset=None):
    if offset is None:
        for contour in c:
            center = (math.ceil(contour.ellipse[0][0]), math.ceil(contour.ellipse[0][1]))
            axes = (math.ceil(contour.ellipse[1][0]), math.ceil(contour.ellipse[1][1]))
            drawed_img = cv2.ellipse(drawed_img, [center, axes, contour.ellipse[2]], color=(0, 255, 0))
    else:
        for contour in c:
            center = (math.ceil(contour.ellipse[0][0])+offset[0], math.ceil(contour.ellipse[0][1])+offset[1])
            axes = (math.ceil(contour.ellipse[1][0]), math.ceil(contour.ellipse[1][1]))
            drawed_img = cv2.ellipse(drawed_img, [center, axes, contour.ellipse[2]], color=(0, 255, 0))
    return drawed_img