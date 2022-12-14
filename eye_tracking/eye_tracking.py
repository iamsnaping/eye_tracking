import os
import threading
from datetime import datetime

import numpy as np
import pynput.keyboard

from base_estimation.plcr import plcr as bp
from pupil_detect import contour_detector_single as cds
import cv2
from pupil_detect import contour_detector_single_debug as cdsd
from eye_utils import data_util as du
import timeit
import pyautogui
import multiprocessing
import queue
import sys
from pynput import keyboard
from data_geter.data_geter import *
from PIL import ImageGrab
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF,Matern,PairwiseKernel,ConstantKernel

STOP_FLAG = False
EXIT_FLAG = False


def on_press(key):
    if isinstance(key, pynput.keyboard.KeyCode):
        if key.char == 'q':
            global EXIT_FLAG
            EXIT_FLAG = True
            return False


q = queue.Queue(30)

root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.26\\wtc1\\cali'
root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.11.8\\wtc\\test\\33'

root_path = 'C:\\Users\\snapping\\Desktop\\'
pic_path = os.path.join(root_path, '0.png')
img = cv2.imread(pic_path)

params = cds.PuRe_params()
params.threshold1 = 30
params.threshold2 = 60
params.r_th = 0.3
params.find_contour_param = cv2.CHAIN_APPROX_NONE
params.gd_max = 20
params.gd_min = 1
params.glints_num = 5
params.pd_min = 10
params.pd_max = 50
params.g_threshold = 30
params.p_binary_threshold = 80
params.g_binary_threshold = 140

time_recoder = np.array([0., 0.])
times_recorder = np.array([0., 0.])

p_cali_vec=np.array([[ 26.39,-34.26],[-26.39,-34.26],[-26.39,-2.999998],[26.39,-2.999998]],dtype=np.float32)

def get_pics(cap):
    while True:
        ref, frame = cap.read()
        if STOP_FLAG:
            break
        if not q.full():
            q.put(frame)


def clear_q(t):
    global STOP_FLAG
    STOP_FLAG = True
    t.join()
    print(q.qsize())
    while not q.empty():
        q.get()


def record_time(type):
    def decorator(func):
        def inner():
            global time_recoder
            start = timeit.default_timer()
            func()
            end = timeit.default_timer()
            time_recoder[type] += end - start
            return func

        return inner

    return decorator


# nums=[19,2,21,23,28,29,3,30,5]
nums = [303]
skip_nums = [19]
#

debug_detector = cdsd.PuRe(params)
# debug_detector=cds.PuRe(params)
if __name__ == '__main__':
    for i in nums:
        if i in skip_nums:
            continue
        img_path = os.path.join(root_path, str(i) + '.png')
        print(img_path)
        origin_img = cv2.imread(img_path)
        gray_img = du.get_gray_pic(origin_img)
        res = debug_detector.detect(gray_img)
        if isinstance(res, bool):
            continue
        img = cdsd.draw_ellipse(origin_img, res[0][0])
        img = cdsd.draw_ellipse(img, res[1][0])
        img = cdsd.draw_ellipse(img, [res[0][1]])
        img = cdsd.draw_ellipse(img, [res[1][1]])
        for i in res[0][0]:
            print(i.ellipse, end=' ')
        print('')
        for i in res[1][0]:
            print(i.ellipse, end=' ')
        du.show_ph(img, name=str(i))
        # breakpoint()
    breakpoint()


'''
1 0
2  3  g0 
'''

def get_glints_sort(glints):
    glints.sort(key=lambda x: x[1])
    sub_glints = glints[2:5]
    sub_glints.sort(key=lambda x: x[0])
    if glints[0][0] < glints[1][0]:
        glints[0], glints[1] = glints[1], glints[0]
    glints[2], glints[3], glints[4] = sub_glints[0], sub_glints[2], sub_glints[1]


# 左右眼针对 屏幕上显示的左右眼
# 初始为非镜像操作，即左眼对应屏幕右眼，右眼对应屏幕左眼
#trend mean25.165742874145508 truth mean 26.813568115234375
#trend mean23.451723098754883 truth mean 26.099292755126953
# 左眼减去右眼得到眼部的向量
class eye_tracker:
    def __init__(self):
        params = cds.PuRe_params()
        self.is_calibration = True
        params.threshold1 = 30
        params.threshold2 = 60
        params.r_th = 0.3
        params.find_contour_param = cv2.CHAIN_APPROX_NONE
        params.gd_max = 20
        params.gd_min = 1
        params.glints_num = 5
        params.pd_min = 10
        params.pd_max = 50
        params.g_threshold = 30
        params.p_binary_threshold = 80
        params.g_binary_threshold = 140
        self.average_mode = False
        self.pure = cds.PuRe(params)
        self.plcr = [bp.plcr(52.78, 31.26), bp.plcr(52.78, 31.26)]
        self.plcr[0]._rt = 220
        self.plcr[0]._radius = 0.78
        self.plcr[1]._rt = 220
        self.plcr[1]._radius = 0.78
        # 初始头部位置 第一个头部修正
        self.origin_position = np.zeros(2, dtype=np.float32)
        # 初始 头部的左眼右眼
        self.oleft_eye = np.zeros(2, dtype=np.float32)
        self.oright_eye = np.zeros(2, dtype=np.float32)
        # 头部的左眼右眼
        self.left_eye = np.zeros(2, dtype=np.float32)
        self.right_eye = np.zeros(2, dtype=np.float32)
        # 头部旋转中心距眼部连线的直线距离
        self.head_radius = 8.
        # 左右旋转角度
        self.lrrotation = np.float32(0.)
        # 上下旋转角度
        self.udrotation = np.float32(0.)
        # 水平偏移
        self.head_offset = np.float32(0.)
        self._s=np.float32(0.)
        self.gpr_l=0.
        self.gpr_r=0.
        self.gpr_c=0.

    def set_params(self, params):
        self.pure.set_params(params)

    def set_d(self, x, y):
        self.plcr[0]._rt = x * 4
        self.plcr[1]._rt = y * 4

    def set_calibration(self, vec, des):
        if len(vec) == 2:
            self.plcr[0]._is_calibration = False
            self.plcr[1]._is_calibration = False
            self.plcr[0].set_calibration(vec[0], des[0])
            self.plcr[1].set_calibration(vec[1], des[1])
            t1=0
            t2=0
            for i in range(1,100):
                kernel = ConstantKernel(1.) *DotProduct(sigma_0=float(i)/100.)+ConstantKernel(0.9)*WhiteKernel()
                g = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(des[0], vec[0])
                kernel = ConstantKernel(1.) * DotProduct(sigma_0=float(i)/100) + ConstantKernel(0.9) *WhiteKernel()
                g2 = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(des[1], vec[1])
                s1=g.score(des[0], vec[0])
                s2=g2.score(des[1],vec[1])
                if s1>t1:
                    t1=s1
                    self.plcr[0].gpr = g
                if s2>t2:
                    t2=s2
                    self.plcr[1].gpr = g2
        else:
            self.plcr[0]._is_calibration = True
            self.plcr[1]._is_calibration = True
            self.average_mode = True
            self.is_calibration = False
            self.plcr[0].set_calibration(vec[0], des[0])
            self.plcr[1].set_calibration(vec[0], des[0])
            self.vecs = vec[0]
            self.des = des[0]
            t=0
            for i in range(1,100):

                kernel  = ConstantKernel(1.) *DotProduct(sigma_0=0.99)+ConstantKernel(1.)*WhiteKernel()
                g = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(des[0], vec[0])
                s=g.score(des[0],vec[0])
                if s>t:
                    t=s
                    self.gpr = g

    def get_estimation(self, num, eye_num):
        self.plcr[eye_num]._pupil_center = np.array([num[0][0], num[0][1], 0]).reshape((3, 1))
        self.plcr[eye_num]._param = np.array([0, 0, 0.62], dtype=np.float32).reshape((3, 1))
        self.plcr[eye_num].get_param()
        self.plcr[eye_num]._up = np.array([0, 1, 0], dtype=np.float32).reshape((3, 1))
        light = np.array(
            [num[1][0], num[1][1], 0, num[2][0], num[2][1], 0, num[3][0], num[3][1], 0, num[4][0], num[4][1], 0],
            dtype=np.float32).reshape((4, 3))
        light = light.T
        # 以瞳孔中心建立坐标
        self.plcr[eye_num]._glints = self.plcr[eye_num]._pupil_center - light
        self.plcr[eye_num]._g0 = np.array([num[5][0], num[5][1], 0], dtype=np.float32).reshape((3, 1))
        self.plcr[eye_num]._g0 = self.plcr[eye_num]._pupil_center - self.plcr[eye_num]._g0
        # self.plcr[eye_num].get_e_coordinate()
        # self.plcr[eye_num].transform_e_to_i()
        # self.plcr[eye_num].get_plane()
        # self.plcr[eye_num].get_visual()
        self.plcr[eye_num].get_m_points()
        return self.plcr[eye_num].gaze_estimation()

    def calibration_mode(self, mode):
        self.is_calibration = mode
        self.plcr[0]._is_calibration = mode
        self.plcr[1]._is_calibration = mode

    def head_calibration(self):
        vec = self.left_eye - self.right_eye
        o_vec = self.oleft_eye - self.oright_eye
        if norm(o_vec)==0:
            return
        lrrotation = norm(vec) / norm(o_vec)
        vec_norm=norm(vec)
        vec /= vec_norm
        o_vec = o_vec / norm(o_vec)
        udrotation = np.vdot(vec, o_vec)
        sub = (self.left_eye + self.right_eye) / 2
        self.head_offset = np.array([self.origin_position[0] - sub[0], sub[1] - self.origin_position[1]], dtype=np.float32)
        self._s=1920/(26.39*(self.plcr[0]._s+self.plcr[1]._s))
        self.head_offset*=self._s

    # binary img
    def detect(self, img, img_p=None, f=False):
        global time_recoder, times_recorder
        time1 = timeit.default_timer()
        res = self.pure.detect(img=img)
        time2 = timeit.default_timer()
        times_recorder[0] += 1.
        time_recoder[0] += time2 - time1
        if isinstance(res, bool):
            return 'can not detect glints or pupils'
        self.left_eye=self.pure.left_eye
        self.right_eye=self.pure.right_eye
        glints_l = []
        glints_r = []
        # sub_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(res[0][0]) < 5 or len(res[1][0]) < 5:
            return 'glints is not enough'
        df = True
        if len(res[0][0]) == 5:
            for c in res[0][0]:
                glints_l.append(c.ellipse[0])

            df = False
            '''
            draw_img = cds.draw_ellipse(sub_img, res[0][0])
            draw_img = cds.draw_ellipse(draw_img, [res[0][1]])
            draw_img = cv2.circle(draw_img, (int(res[0][1].ellipse[0][0]), int(res[0][1].ellipse[0][1])), 3,(0, 255, 0), -1)
            '''
        if len(res[1][0]) == 5:
            for c in res[1][0]:
                glints_r.append(c.ellipse[0])
        '''
            if df:
                draw_img = cds.draw_ellipse(sub_img, res[0][0])
                draw_img = cds.draw_ellipse(draw_img, [res[0][1]])
                draw_img = cv2.circle(draw_img, (int(res[1][1].ellipse[0][0]), int(res[1][1].ellipse[0][1])), 3,(0, 255, 0), -1)
            else:
                draw_img = cds.draw_ellipse(draw_img, res[1][0])
                draw_img = cds.draw_ellipse(draw_img, [res[1][1]])
                draw_img = cv2.circle(draw_img, (int(res[1][1].ellipse[0][0]), int(res[1][1].ellipse[0][1])), 3,(0, 255, 0), -1)
        cv2.imwrite(img_p,draw_img)
        if len(res[0][0]) < 5 or len(res[1][0]) < 5:
            return 'glints is not enough'
        '''
        if len(glints_r) == 0:
            get_glints_sort(glints_l)
            glints_r = glints_l.copy()
            left = [res[0][1].ellipse[0]]
            right = left.copy()
        elif len(glints_l) == 0:
            get_glints_sort(glints_r)
            glints_l = glints_r.copy()
            right = [res[1][1].ellipse[0]]
            left = right.copy()
        else:
            get_glints_sort(glints_l)
            get_glints_sort(glints_r)
            left = [res[0][1].ellipse[0]]
            right = [res[1][1].ellipse[0]]
        '''
        file_path=os.path.dirname(img_p)+'\\'+os.path.basename(img_p).split('.')[0]+'.txt'
        with open(file_path,'w') as t:
            for ce in glints_l:
                t.write(str(ce)+'\n')
            t.write(str(res[0][1].ellipse[0])+'\n\n')
            for ce in glints_r:
                t.write(str(ce)+'\n')
            t.write(str(res[1][1].ellipse[0]))
        '''
        # if f:
        #     self.oright_eye=self.right_eye
        #     self.oleft_eye=self.left_eye
        #     self.origin_position=(self.left_eye+self.right_eye)/2
        # for i in range(4):
        #     glints_l[i]+=(1-np.sqrt(1-0.78/8**2))/65*p_cali_vec[i]*self.plcr[0]._s
        #     glints_r[i]+= (1-np.sqrt(1-0.78/8**2))/65*p_cali_vec[i]*self.plcr[1]._s

        self.head_calibration()
        self.plcr[0].refresh()
        self.plcr[1].refresh()
        left.extend(glints_l)
        right.extend(glints_r)
        l = self.get_estimation(left, 0)
        r = self.get_estimation(right, 1)
        g_len = (l[0] + r[0]) / 2
        # print(f'glen {g_len}')
        gaze_estimation = l * (1920 - g_len) / 1920 + r * g_len / 1920
        r+=np.array([self.oright_eye[0]-self.right_eye[0],self.right_eye[1]-self.oright_eye[1]],np.float32)*self._s
        l+=np.array([self.oleft_eye[0]-self.left_eye[0],self.left_eye[1]-self.oleft_eye[1]],dtype=np.float32)*self._s
        # print(f'left offset {np.array([self.oleft_eye[0]-self.left_eye[0],self.left_eye[1]-self.oleft_eye[1]],dtype=np.float32)*self._s} {self.oleft_eye} {self.left_eye}')
        # print(f'right offset {np.array([self.oright_eye[0]-self.right_eye[0],self.right_eye[1]-self.oright_eye[1]],np.float32)*self._s} {self.oright_eye} {self.right_eye}')
        # vec direction->down
        # up down left right
        time3 = timeit.default_timer()
        if self.average_mode:
            if self.is_calibration:
                return gaze_estimation+self.head_offset
            else:
                # print('2')
                # print(self.gpr.predict([gaze_estimation],return_std=True)[0][0],gaze_estimation)
                return -self.gpr.predict([gaze_estimation],return_std=True)[0][0]+self.head_offset+gaze_estimation
            centers = np.array([self.des[0][0], self.des[0][1]], dtype=np.float32)
            # y 480,960 x 660 1320
            # up
            vec1 = (self.vecs[0] - self.vecs[2])
            # down
            vec2 = (self.vecs[5] - self.vecs[0])
            # up
            vec3 = (self.vecs[4] - self.vecs[1])
            vec4 = (self.vecs[6] - self.vecs[3])
            centers_ratio = norm(self.des[0] - self.des[2]) / (
                    norm(self.des[5] - self.des[0]) + norm(self.des[0] - self.des[2]))
            if centers[0] > gaze_estimation[0]:
                left_vec = self.des[1] * (1 - centers_ratio) + self.des[4] * centers_ratio
                ratio = norm(gaze_estimation - self.des[1]) / (
                        norm(self.des[2] - gaze_estimation) + norm(gaze_estimation - self.des[1]))
                mid1 = ratio * self.des[2] + (1 - ratio) * self.des[1]
                ratio = norm(gaze_estimation - left_vec) / (
                        norm(gaze_estimation - left_vec) + norm(gaze_estimation - centers))
                mid2 = ratio * self.des[0] + (1 - ratio) * left_vec
                ratio = norm(gaze_estimation - self.des[4]) / (
                        norm(gaze_estimation - self.des[4]) + norm(gaze_estimation - self.des[5]))
                mid3 = ratio * self.des[5] + (1 - ratio) * self.des[4]
                scal_1 = norm(gaze_estimation - mid1) / norm(mid3 - mid1)
            else:
                right_vec = self.des[3] * (1 - centers_ratio) + self.des[6] * centers_ratio
                ratio = norm(gaze_estimation - self.des[2]) / (
                        norm(self.des[2] - gaze_estimation) + norm(gaze_estimation - self.des[3]))
                mid1 = ratio * self.des[3] + (1 - ratio) * self.des[2]
                ratio = norm(gaze_estimation - centers) / (
                        norm(gaze_estimation - right_vec) + norm(gaze_estimation - centers))
                mid2 = ratio * right_vec + (1 - ratio) * centers
                ratio = norm(gaze_estimation - self.des[5]) / (
                        norm(gaze_estimation - self.des[5]) + norm(gaze_estimation - self.des[6]))
                mid3 = ratio * self.des[6] + (1 - ratio) * self.des[5]
                scal_1 = np.linalg.norm(gaze_estimation - mid1) / np.linalg.norm(mid3 - mid1)
            if gaze_estimation[1] < mid2[1]:
                scal_2 = np.linalg.norm(gaze_estimation - mid1) / np.linalg.norm(mid2 - mid1)
                vec5 = scal_2 * vec1 + self.vecs[2]
                vec5_des = self.des[2] + (self.des[0] - self.des[2]) * scal_2
            else:
                scal_2 = np.linalg.norm(gaze_estimation - mid2) / np.linalg.norm(mid3 - mid2)
                vec5 = scal_2 * vec2 + self.vecs[0]
                vec5_des = self.des[0] + (self.des[5] - self.des[0]) * scal_2
            if gaze_estimation[0] < centers[0]:
                vec6 = vec3 * scal_1 + self.vecs[1]
                vec6_des = (self.des[4] - self.des[1]) * scal_1
                # compute_vec = (vec5 - vec6) / (660 * s_para) * gaze_estimation[0] + vec6
                compute_vec = (vec5 - vec6) / np.linalg.norm(vec6_des[0] - vec5_des[0]) * gaze_estimation[0] + vec6
            else:
                vec6 = vec4 * scal_1 + self.vecs[3]
                vec6_des = (self.des[6] - self.des[3]) * scal_1
                # compute_vec = (vec6 - vec5) / (660 * s_para) * (gaze_estimation[0] - centers[0]) + vec5
                compute_vec = (vec6 - vec5) / np.linalg.norm(vec6_des[0] - vec5_des[0]) * (
                        gaze_estimation[0] - centers[0]) + vec5
            # print(f'sca {scal_1, scal_2,mid1,mid2,mid3,gaze_estimation}')
            time3 = timeit.default_timer()
            time_recoder[1] += time3 - time2
            times_recorder[1] += 1.
            return gaze_estimation - compute_vec+self.head_offset
        times_recorder[1] += 1
        time_recoder[1] += time3 - time2
        return l, r

    def calibration(self, cap, screen, points, cali_nums=20, store=False, root_path=None):
        points_num = 1
        vecs_left = []
        vecs_right = []
        vecs_ave = []
        des_left, des_right, des_ave = [], [], []
        background_color = (255, 255, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        r_cali_path = os.path.join(root_path, 'cali')
        r_draw_path = os.path.join(root_path, 'draw')
        points = np.array(points, dtype=np.float32)
        print('begin')
        for point in points:
            nums = 0
            ress = []
            cali_path = os.path.join(r_cali_path, str(points_num))
            draw_path = os.path.join(r_draw_path, str(points_num))
            if not os.path.exists(cali_path):
                os.makedirs(cali_path)
            if not os.path.exists(draw_path):
                os.makedirs(draw_path)
            screen.fill(background_color)
            pygame.draw.circle(screen, RED, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            temp_centers = []
            while True:
                pygame.draw.circle(screen, BLUE, (point[0], point[1]), 10)
                pygame.display.update()
                if nums >= cali_nums:
                    break
                ref, frame = cap.read()
                gray_img = du.get_gray_pic(frame)
                d_path = os.path.join(draw_path, str(nums) + '.png')
                res = self.detect(gray_img, img_p=d_path,f=(points_num==points_num))
                if isinstance(res, str):
                    continue
                self.left_eye=self.pure.left_eye
                self.right_eye=self.pure.right_eye
                temp_centers.append([self.pure.left_eye,self.pure.right_eye])
                ress.append(np.array(res, dtype=np.float32))
                nums += 1
                print(nums)
            pygame.draw.circle(screen, GREEN, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            res = []
            _ = len(ress)
            centers=[]
            for i in range(_):
                if i == 0:
                    left = np.linalg.norm(ress[0] - ress[_ - 1])
                    right = np.linalg.norm(ress[i] - ress[i + 1])
                elif i == _ - 1:
                    left = np.linalg.norm(ress[i] - ress[i - 1])
                    right = np.linalg.norm(ress[i] - ress[0])
                else:
                    left = np.linalg.norm(ress[i] - ress[i - 1])
                    right = np.linalg.norm(ress[i] - ress[i + 1])
                if left > 400 and right > 400:
                    continue
                centers.append(temp_centers[i])
                res.append(ress[i])
            eye_center=np.mean(centers,axis=0)
            self.left_eye=eye_center[0]
            self.right_eye=eye_center[1]
            sub = (self.left_eye + self.right_eye) / 2
            c_offset = np.array([self.origin_position[0] - sub[0], sub[1] - self.origin_position[1]],
                                dtype=np.float32) * self._s
            if points_num == 1:
                self.origin_position = eye_center.mean(axis=0)
                self.oleft_eye = eye_center[0]
                self.oright_eye = eye_center[1]
                origin_position = self.origin_position
                oleft = eye_center[0]
                oright = eye_center[1]
            # print(difference)
            if len(res) == 0:
                continue
            res = np.mean(res, axis=0)
            g_len = np.fabs((res[0][0] + res[1][0]) / 2)
            vecs_right.append((res[1] - point + np.array(
                [oright[0] - self.right_eye[0], self.right_eye[1] - oright[1]], np.float32) * self._s))
            vecs_left.append(
                (res[0] - point) + np.array([oleft[0] - self.left_eye[0], self.left_eye[1] - oleft[1]],
                                            dtype=np.float32) * self._s)
            vecs_ave.append((res[0] * (1920. - g_len) / 1920.) + (res[1] * (g_len) / 1920) - point + c_offset)
            des_left.append(res[0] + np.array([oleft[0] - self.left_eye[0], self.left_eye[1] - oleft[1]],
                                              dtype=np.float32) * self._s)
            des_right.append(res[1] + np.array([oright[0] - self.right_eye[0], self.right_eye[1] - oright[1]],
                                               np.float32) * self._s)
            des_ave.append((res[0] * (1920. - g_len) / 1920.) + (res[1] * (g_len) / 1920) - c_offset)
            print(f'point {points_num} finished')
            points_num+=1
        pygame.quit()
        self.origin_position=origin_position
        self.oleft_eye=oleft
        self.oright_eye=oright
        return np.array(vecs_left, dtype=np.float32), \
               np.array(vecs_right, dtype=np.float32), \
               np.array(vecs_ave, dtype=np.float32), \
               np.array(des_left, dtype=np.float32), np.array(des_right, dtype=np.float32), np.array(des_ave,
                                                                                                     dtype=np.float32)

    def tracking(self, cap, vec_left, vec_right, vec_ave, des_left, des_right, des_ave, store=False, root_path=None):
        pyautogui.FAILSAFE = False
        name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 当前的时间（当文件名）
        screen = ImageGrab.grab()  # 获取当前屏幕
        width, high = screen.size  # 获取当前屏幕的大小
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # MPEG-4编码,文件后缀可为.avi .asf .mov等
        video = cv2.VideoWriter('%s.avi' % name, fourcc, 15, (width, high))  # （文件名，编码器，帧率，视频宽高）
        # print('3秒后开始录制----')  # 可选
        # time.sleep(3)
        global start_time
        start_time = time.time()

        def run_key():
            with keyboard.Listener(on_press=on_press) as lsn:
                lsn.join()

        t = threading.Thread(target=run_key)
        t.start()
        if not (root_path is None):
            if os.path.exists(root_path):
                os.makedirs(root_path)
        self.set_calibration([vec_ave], [des_ave])
        # tracker_2.set_calibration([np.array([[0.,0.] for i in range(7)])],[np.array([[0.,0.] for i in range(7)])])
        start_t = timeit.default_timer()
        point = np.zeros(2, dtype=np.float32)
        times = 0.
        valid = 0.
        data_t = 0.
        pro_t = 0.
        # thread_p=threading.Thread(target=get_pics,args=(cap,))
        # thread_p.setDaemon(True)
        # thread_p.start()
        global EXIT_FLAG
        print('开始录制!')
        while True:
            if EXIT_FLAG == True:
                break
            # print('a')
            im = ImageGrab.grab()
            if q.empty():
                t1 = timeit.default_timer()
                ref, frame = cap.read()
                # print(frame.shape)
                # frame=q.get()
                # print(frame)
                t2 = timeit.default_timer()
                gray_img = du.get_gray_pic(frame)
                res = self.detect(gray_img)
                t3 = timeit.default_timer()
                pro_t += t3 - t2
                data_t += t2 - t1
                times += 1
                if not isinstance(res, str):
                    valid += 1
                    u_point = res
                    if np.linalg.norm(u_point - point) > 15:
                        point = u_point
                print(res)
                im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                cv2.circle(im, (int(point[0]), int(point[1])), 10, (0, 0, 255))
                video.write(im)
            # pyautogui.moveTo(point[0], point[1])

            if valid == 50:
                end_t = timeit.default_timer()
                total_t = end_t - start_t
                print(total_t)
                print(valid)
                print(times)
                print(pro_t, pro_t / times, pro_t / valid)
                print(data_t, data_t / times, data_t / valid)
                print(pro_t / total_t)
                print(data_t / total_t)
                # clear_q(thread_p)
                # print(q.qsize())
                # break


def norm(a):
    return np.linalg.norm(a)


def get_time():
    a = cds.get_time()
    return a, time_recoder, times_recorder, time_recoder / times_recorder, time_recoder / time_recoder.sum()
