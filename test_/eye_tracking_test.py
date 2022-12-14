import os
import random
import timeit

import test_main as tm
import cv2
import numpy as np
import pygame
import ctypes
from eye_tracking import eye_tracking as et
from eye_utils import data_util as du
from data_geter.data_geter import *
import Cython
from Cython.Build import cythonize
from matplotlib import pyplot as plt
from goto import with_goto


root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.11.8\\wtc'
pic_path = os.path.join(root_path, 'drawed')
if not os.path.exists(pic_path):
    os.makedirs(pic_path)
bias_num = 5

COLORS=['purple','green','blue','pink','brown','red','lightblue','teal','orange','lightgreen','magenta','yellow','sky blue','grey','limegreen','lightpurple','violet','darkgreen']

cali_right_vec = np.array([[-1.22, -8.13], [0.81, -10.42], [6.97, -6.55], [8.41, 2.13], [-0.09, 0.52], [-1.78, 1.15]],
                          dtype=np.float32)
cali_left_vec = np.array([[-5.64, -8.93], [-1.46, -10.95], [3.62, -9.4], [4.35, -2.51], [-2.44, 2.03], [-7.46, -1.61]],
                         dtype=np.float32)
cali_ave_vec = np.array([[-1.62, -8.2], [-0.24, -10.67], [3.59, -9.42], [4.24, -2.63], [-0.99, -0.46], [-2.15, 0.96]],
                        dtype=np.float32)

read_img = 0.
img_times = 0.
read_txt = 0.
txt_times = 0.


def get_path(dir):
    global root_path
    r_path = os.path.join(root_path, dir)
    d_path = os.path.join(pic_path, dir)
    dirs_list = os.listdir((r_path))
    dirs_list.sort(key=lambda x: (len(x), x))
    t_files = []
    i_files = []
    images_files = []
    for dirs in dirs_list:
        t_file = []
        i_file = []
        images = []
        dirs_p = os.path.join(r_path, dirs)
        draw_path = os.path.join(d_path, dirs)
        if not os.path.exists(draw_path):
            os.makedirs(draw_path)
        dir = os.listdir(dirs_p)
        dir.sort(key=lambda x: (len(x), x))
        for d in dir:
            if os.path.splitext(d)[1] == '.txt':
                t_file.append(os.path.join(dirs_p, d))
            else:
                images.append(os.path.join(draw_path, d))
                i_file.append(os.path.join(dirs_p, d))
        t_files.append(t_file)
        i_files.append(i_file)
        images_files.append(images)
    return t_files, i_files, images_files

@with_goto
def main():
    global read_img, img_times,txt_times,read_txt
    time1 = timeit.default_timer()
    ttxt = []
    vec_left, vec_right, vec_ave, des_left, des_right, des_ave,oleft,oright,oposition = cali()
    time2 = timeit.default_timer()
    # print('it is cali')
    print(vec_left,vec_right,vec_ave,des_left,des_right,des_ave)
    tracker_1 = et.eye_tracker()
    # tracker_1.set_calibration([np.array([[0.,0.] for i in range(7)]),np.array([[0.,0.] for i in range(7)])],[np.array([[0.,0.] for i in range(7)]),np.array([[0.,0.] for i in range(7)])])
    tracker_1.set_calibration([vec_left, vec_right], [des_left, des_right])
    tracker_2 = et.eye_tracker()
    tracker_2.set_calibration([vec_ave], [des_ave])
    # tracker_2.set_calibration([np.array([[0.,0.] for i in range(7)])],[np.array([[0.,0.] for i in range(7)])])
    tracker_1.calibration_mode(False)
    # tracker_1.calibration_mode(True)
    # tracker_2.calibration_mode(True)
    tracker_1.origin_position=oposition
    tracker_2.origin_position=oposition
    tracker_1.oleft_eye,tracker_1.oright_eye,tracker_2.oleft_eye,tracker_2.oright_eye=oleft,oright,oleft,oright
    vec_left = np.zeros(2, dtype=np.float32)
    vec_right = np.zeros(2, dtype=np.float32)
    vec_ave = np.zeros(2, dtype=np.float32)
    vec_ave_2 = np.zeros(2, dtype=np.float32)
    left = []
    right = []
    ave = []
    ave_2 = []
    tt = 0
    img_ps = []
    txtss, imgss, imagess = get_path('test')
    max_a = -1
    max_a2 = -1
    points_len = []
    r_truth = []
    r_truth2 = []
    value_np=[]
    era_value=np.array([0. for i in range(9)],dtype=np.float32)
    era_times=np.array([0. for i in range(9)],dtype=np.float32)
    dis_left = []
    dis_right = []
    x1,x2,x3,x4,x5=[],[],[],[],[]
    y1,y2,y3,y4,y5=[],[],[],[],[]
    colors=[]
    for txts, imgs, images in zip(txtss, imgss, imagess):
        ress = []
        ress2 = []
        txt_num = np.zeros((2),dtype=np.float32)
        for txt, img_p, image in zip(txts, imgs, images):
            time3 = timeit.default_timer()
            img = cv2.imread(img_p)
            time4 = timeit.default_timer()
            img_times += 1.
            read_img += time4 - time3
            gray_img = du.get_gray_pic(img)
            res = tracker_1.detect(gray_img, image)
            res2 = tracker_2.detect(gray_img, image)
            with open(txt) as t:
                txt = t.read()
            txt = txt.split(' ')
            txt_num = np.array(txt, dtype=np.float32)
            # print(txt, txt_num)
            if isinstance(res, str):
                # print(f'{img_p} pass')
                continue
            time5 = timeit.default_timer()
            time6 = timeit.default_timer()
            read_txt += time6 - time5
            txt_times += 1.
            ress.append(np.array(res,dtype=np.float32))
            ress2.append(np.array(res2,dtype=np.float32))
        # print(txt_num)
        x1.append(txt_num[0])
        x2.append(txt_num[0])
        x3.append(txt_num[0])
        x4.append(txt_num[0])
        y1.append(txt_num[1])
        y2.append(txt_num[1])
        y3.append(txt_num[1])
        y4.append(txt_num[1])
        colors.append(0)
        if len(ress) == 0 or len(ress2) == 0:
            # print(f'{img_p} pass')
            continue
        if isinstance(txt, str):
            # print(f'{img_p} pass')
            continue
        # average = np.mean(ress, axis=0)
        # average2 = np.mean(ress2, axis=0)
        res = []
        res2 = []
        if len(ress)==1:
            res.append(ress[0])
        if len(ress2)==1:
            res2.append(ress2[0])
        _ = len(ress)
        if _==1:
            goto .a
        for i in range(_):
            if i == 0:
                left2 = np.linalg.norm(ress[0] - ress[_ - 1])
                right2 = np.linalg.norm(ress[0] - ress[1])
            elif i == _ - 1:
                left2 = np.linalg.norm(ress[i] - ress[i - 1])
                right2 = np.linalg.norm(ress[i] - ress[0])
            else:
                left2 = np.linalg.norm(ress[i] - ress[i - 1])
                right2 = np.linalg.norm(ress[i] - ress[i + 1])
            if not (left2 > 400 and right2 > 400):
                res.append(ress[i])
        label .a
        _=len(ress2)
        if _==1:
            goto .begin
        for i in range(_):
            if i == 0:
                left1 = np.linalg.norm(ress2[0] - ress2[_ - 1])
                right1 = np.linalg.norm(ress2[i] - ress2[i + 1])
            elif i == _ - 1:
                left1 = np.linalg.norm(ress2[i] - ress2[i - 1])
                right1 = np.linalg.norm(ress2[i] - ress2[0])
            else:
                left1 = np.linalg.norm(ress2[i] - ress2[i - 1])
                right1 = np.linalg.norm(ress2[i] - ress2[i + 1])
            if not (left1 > 300 and right1 > 300):
                res2.append(ress2[i])
        label .begin
        if len(res2)==0 or len(res)==0:
            continue
        points_len.append(len(ress))
        res2 = np.mean(res2, axis=0)
        res = np.mean(res, axis=0)
        ttxt.append(txt_num)
        # 未融合
        x1.append(res[0][0])
        x2.append(res[1][0])
        y1.append(res[0][1])
        y2.append(res[1][1])
        # 融合
        x3.append(res2[0])
        y3.append(res2[1])
        colors.append(int((txt_num[0]-60)/120+2)*10)
        max_a = max(max_a, np.linalg.norm((res[1] + res[0]) / 2 - txt_num))
        max_a2 = max(max_a2, np.linalg.norm(res2 - txt_num ))
        ave.append((res[1] + res[0]) / 2 - txt_num )
        left.append(res[0] - txt_num )
        right.append(res[1] - txt_num )
        ave_2.append(res2 - txt_num )
        value_np.append(np.linalg.norm(res2 - txt_num ))
        t=int((txt_num[0]//640)*3 + txt_num[1]//360)
        era_value[t]+=np.linalg.norm(res2 - txt_num )
        era_times[t]+=1.
        dis_left.append(res[0])
        dis_right.append(res[1])
        img_ps.append(img_p)
        g_len = np.fabs((res[0][0] + res[1][0]) / 2)
        r_truth.append((res[0] * (1920. - g_len) / 1920.) + (res[1] * g_len /1920.) - txt_num)
        _truth=(res[0] * (1920. - g_len) / 1920.) + (res[1] * g_len /1920.)
        x4.append(_truth[0])
        y4.append(_truth[1])
        imgs.append(image)
        vec_left += (res[0] - txt_num )
        vec_right += (res[1] - txt_num)
        vec_ave += ((res[1] + res[0]) / 2 - txt_num)
        # print(res2,txt_num)
        vec_ave_2 += (res2 - txt_num )
        tt += 1
    x1=np.array(x1)
    x2=np.array(x2)
    x3=np.array(x3)
    y1=np.array(y1)
    y2=np.array(y2)
    y3=np.array(y3)
    colors=np.array(colors)
    plt.scatter(x1,y1,c=colors,cmap='viridis')
    plt.colorbar()
    plt.savefig('C:\\Users\\snapping\\Desktop\\left.png')
    # plt.show()
    plt.cla()
    plt.scatter(x2, y2, c=colors,cmap='viridis')
    plt.savefig('C:\\Users\\snapping\\Desktop\\right.png')
    # plt.show()
    plt.cla()
    plt.scatter(x3, y3, c=colors,cmap='viridis')
    plt.savefig('C:\\Users\\snapping\\Desktop\\trend2.png')
    plt.cla()
    plt.scatter(x4,y4,c=colors,cmap='viridis')
    plt.savefig('C:\\Users\\snapping\\Desktop\\truth.png')
    # plt.show()
    vec_left /= tt
    vec_right /= tt
    vec_ave /= tt
    vec_ave_2 /= tt
    user32 = ctypes.windll.user32
    # '''
    dg = data_geter('')
    user32.SetProcessDPIAware(2)
    screen = pygame.display.set_mode((user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)))
    background_color = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    truth_total = []
    trend2_total = []
    pygame.init()
    pygame.display.set_caption("data geter")
    t = 0
    screen.fill(background_color)
    c = 255.0
    points = dg.get_points(20, 1920, 1080)
    for point in points:
        x = point[0] - 60
        y = point[1] - 60
        w, h = 120, 120
        # c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
    number_front = pygame.font.SysFont(None, 30)
    number_front2 = pygame.font.SysFont(None, 30)
    for t, a, pl in zip(ttxt, ave, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'trend1.png'))
    screen.fill(background_color)
    for point in points:
        x = point[0] - 60
        y = point[1] - 60
        w, h = 120, 120
        # c = int(np.linalg.norm(a) * 255.0 / max_a)
        # color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
    for t, a, pl in zip(ttxt, ave_2, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'trend2.png'))

    screen.fill(background_color)
    for point in points:
        x = point[0] - 60
        y = point[1] - 60
        w, h = 120, 120
        # c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
    for t, a, pl in zip(ttxt, value_np, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        trend2_total.append(a)
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'trend2_1.png'))
    screen.fill(background_color)
    for t, a, pl in zip(ttxt, left, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'left.png'))
    screen.fill(background_color)
    for t, a, pl in zip(ttxt, right, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'right.png'))
    screen.fill(background_color)
    for t, a, pl in zip(ttxt, r_truth, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'truth.png'))
    screen.fill(background_color)
    for t, a, pl in zip(ttxt, r_truth, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        truth_total.append(np.linalg.norm(a))
        number_txt = str(np.round(np.linalg.norm(a), 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'truth_2.png'))
    screen.fill(background_color)
    for t, a, pl in zip(ttxt, dis_left, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'dis_right.png'))
    screen.fill(background_color)
    for t, a, pl in zip(ttxt, dis_right, points_len):
        x = t[0] - 60
        y = t[1] - 60
        w, h = 120, 120
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
        rec1 = pygame.Rect(x, y, 120, 120)
        pygame.draw.rect(screen, (0, 0, 0), rec1, 1)
        # number_txt=str(int(np.linalg.norm(a)))
        number_txt = str(np.round(a, 2))
        number_txt2 = str(pl)
        number_img = number_front.render(number_txt, True, (0, 0, 0), (255, 255, 255))
        number_img2 = number_front2.render(number_txt2, True, (0, 0, 0), (255, 255, 255))
        margin_x = (120 - 1 - number_img.get_width()) // 2
        margin_y = (120 - 1 - number_img.get_height()) // 2
        screen.blit(number_img, (x + 2 + margin_x, y + 2 + margin_y))
        screen.blit(number_img2, (x, y))
    pygame.image.save(screen, os.path.join(root_path, 'dis_left.png'))
    # '''
    # print(f'cali {time2 - time1}')
    # print(f'{timeit.default_timer() - time1}')
    # print(f'{read_img} {read_img / img_times}')
    # print(f'{read_txt} {read_txt/txt_times}')
    # gt=et.get_time()
    # t=-4
    # for g in gt:
    #     if isinstance(g,list):
    #         # print('tuple')
    #         for i in g:
    #             print(f'{t} {i}')
    #             t+=1
    #     else:
    #         print(g)
    # print(f'-----------------------------max {max_a} {max_a2}')
    # print('{:<5}{:<30}\t{:<30}\t{:<30}\t{:<30}'.format('', '左眼误差', '右眼误差', '两眼平均误差', '平均修正误差'))
    # # print(('左眼误差右眼误差两眼平均误差 平均修正误差'))
    # for i, a, b, c, d, e in zip(range(len(left)), left, right, ave, ave_2, img_ps):
    #     print('{:<3}: {:<30}\t{:<30}\t{:<30}\t{:<30}\t{}\t'.format(i, str(a), str(b), str(c), str(d), e))
    # print('{:<30}{:<30}{:<30}{:<30}'.format('左眼误差均值', '右眼误差均值', '两眼平均误差均值', '平均修正误差均值'))
    # print('{:<30}\t{:<30}\t{:<30}{:<30}'.format(str(vec_left), str(vec_right), str(vec_ave), str(vec_ave_2)))
    print(f'nine areas {era_value/era_times}')
    print(f'trend mean{np.mean(trend2_total)} truth mean {np.mean(truth_total)}')


def cali():
    global read_img, img_times, txt_times, read_txt
    tracker1 = et.eye_tracker()
    tracker1.set_calibration([np.array([[0., 0.] for i in range(7)]), np.array([[0., 0.] for i in range(7)])],
                             [np.array([[0., 0.] for i in range(7)]), np.array([[0., 0.] for i in range(7)])])
    vecs_left = []
    vecs_right = []
    vecs_ave = []
    count_t=0
    tracker1.calibration_mode(True)
    des_left, des_right, des_ave = [], [], []
    txtss, imgss, imagess = get_path('cali')
    points_num = 1
    origin_position=[]
    oleft=[]
    oright=[]
    for txts, imgs, images in zip(txtss, imgss, imagess):
        nums = 0
        ress = []
        txt_num = np.zeros((2))
        temp_centers = []
        for txt, img_p, image in zip(txts, imgs, images):
            img = cv2.imread(img_p)
            gray_img = du.get_gray_pic(img)
            with open(txt) as t:
                txt = t.read()
            txt = txt.split(' ')
            txt_num = np.array(txt, dtype=np.float32)
            res = tracker1.detect(gray_img,f=(points_num==points_num))
            if isinstance(res, str):
                continue
            tracker1.left_eye = tracker1.pure.left_eye
            tracker1.right_eye = tracker1.pure.right_eye
            temp_centers.append([tracker1.pure.left_eye, tracker1.pure.right_eye])
            ress.append(np.array(res, dtype=np.float32))
            nums += 1
        count_t += 1
        res = []
        _ = len(ress)
        centers = []
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
                # if points_num==7:
                    # print(left,right)
                continue
            # print(f'point {points_num} pass')
            centers.append(temp_centers[i])
            res.append(ress[i])
        eye_center = np.mean(centers, axis=0)
        tracker1.left_eye = eye_center[0]
        tracker1.right_eye = eye_center[1]
        if points_num == 1:
            tracker1.origin_position = eye_center.mean(axis=0)
            tracker1.oleft_eye = eye_center[0]
            tracker1.oright_eye = eye_center[1]
            origin_position=tracker1.origin_position
            oleft=eye_center[0]
            oright=eye_center[1]

        # print(difference)
        sub = (tracker1.left_eye + tracker1.right_eye) / 2
        c_offset = np.array([origin_position[0] - sub[0], sub[1] - origin_position[1]], dtype=np.float32)*tracker1._s
        if len(res) == 0:
            continue
        res = np.mean(res, axis=0)
        point=txt_num
        # print(f'cali c_off {c_offset}')
        g_len = np.fabs((res[0][0] + res[1][0])/2)
        vecs_right.append((res[1] - point + np.array([oright[0]-tracker1.right_eye[0],tracker1.right_eye[1]-oright[1]],np.float32)*tracker1._s))
        vecs_left.append((res[0] - point) + np.array([oleft[0]-tracker1.left_eye[0],tracker1.left_eye[1]-oleft[1]],dtype=np.float32)*tracker1._s)
        vecs_ave.append((res[0] * (1920. - g_len) / 1920.) + (res[1] * (g_len) / 1920) - point + c_offset)
        des_left.append(res[0] + np.array([oleft[0]-tracker1.left_eye[0],tracker1.left_eye[1]-oleft[1]],dtype=np.float32)*tracker1._s)
        des_right.append(res[1] +np.array([oright[0]-tracker1.right_eye[0],tracker1.right_eye[1]-oright[1]],np.float32)*tracker1._s)
        des_ave.append((res[0] * (1920. - g_len) / 1920.) + (res[1] * (g_len) / 1920)-c_offset)
        print(f'point {points_num} finished')
        points_num += 1
    return np.array(vecs_left, dtype=np.float32), \
           np.array(vecs_right, dtype=np.float32), \
           np.array(vecs_ave, dtype=np.float32), \
           np.array(des_left, dtype=np.float32), np.array(des_right, dtype=np.float32), np.array(des_ave,
                                                                                                dtype=np.float32),\
            oleft,oright,origin_position


def test():
    dg = data_geter('C:\\Users\\snapping\\Desktop\\data\\2022.11.12\\wtc')
    points = dg.get_cali_points()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,2)
    cap.set(3, 1920)
    cap.set(4, 1080)
    cvv=cv2.VideoWriter()
    cap.set(6, cvv.fourcc('M', 'J', 'P', 'G'))
    while not cap.isOpened():
        print(cap.isOpened())
        ...
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("data geter")
    tracker_1 = et.eye_tracker()
    tracker_1.calibration_mode(True)
    calis=tracker_1.calibration(cap=cap, screen=screen, root_path='C:\\Users\\snapping\\Desktop\\data\\2022.11.12\\wtc',
                          cali_nums=20, store=True, points=points)
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    pygame.display.set_caption("data geter")
    print(calis)
    points=dg.get_points(10)
    screen.fill((255,255,255))
    np.random.shuffle(points)
    ps = points[0:20]
    for p in ps:
        pygame.dqraw.circle(screen, (255, 0, 0), (p[0], p[1]), 10)
        pygame.display.update()
    # calis=[]
    # for i in range(7):
    #     calis.append(np.array([[0.,0.] for i in range(7)],dtype=np.float32))
    # calis=tuple(calis)
    tracker_1.tracking(cap,*calis)



if __name__ == '__main__':
    test()
    # main()