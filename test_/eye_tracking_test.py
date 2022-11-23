
import os
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




root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.11.8\\wtc'
pic_path = os.path.join(root_path, 'drawed')
if not os.path.exists(pic_path):
    os.makedirs(pic_path)
bias_num = 5

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


def main():
    global read_img, img_times,txt_times,read_txt
    time1 = timeit.default_timer()
    ttxt = []
    vec_left, vec_right, vec_ave, des_left, des_right, des_ave = cali()
    time2 = timeit.default_timer()
    # print('it is cali')
    # print(vec_left,vec_right,vec_ave,des_left,des_right,des_ave)
    tracker_1 = et.eye_tracker()
    # tracker_1.set_calibration([np.array([[0.,0.] for i in range(7)]),np.array([[0.,0.] for i in range(7)])],[np.array([[0.,0.] for i in range(7)]),np.array([[0.,0.] for i in range(7)])])
    tracker_1.set_calibration([vec_left, vec_right], [des_left, des_right])
    tracker_2 = et.eye_tracker()
    tracker_2.set_calibration([vec_ave], [des_ave])
    # tracker_2.set_calibration([np.array([[0.,0.] for i in range(7)])],[np.array([[0.,0.] for i in range(7)])])
    tracker_1.calibration_mode(False)
    tracker_2.calibration_mode(False)
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
            if isinstance(res, str):
                # print(f'{img_p} pass')
                continue
            time5 = timeit.default_timer()
            with open(txt) as t:
                txt = t.read()
            txt = txt.split(' ')
            txt_num = np.array(txt, dtype=np.float32)
            time6 = timeit.default_timer()
            read_txt += time6 - time5
            txt_times += 1.
            ress.append(res)
            ress2.append(res2)
        if len(ress) == 0 or len(ress2) == 0:
            # print(f'{img_p} pass')
            continue
        if isinstance(txt, str):
            # print(f'{img_p} pass')
            continue
        average = np.mean(ress, axis=0)
        average2 = np.mean(ress2, axis=0)
        res = []
        res2 = []
        for r1, r2 in zip(ress, ress2):
            if not np.linalg.norm(r1 - average) > 50:
                res.append(r1)
            if not np.linalg.norm(r2 - average2) > 50:
                res2.append(r2)
        if len(res) == 0 or len(res2) == 0:
            continue
        points_len.append(len(ress))
        res2 = np.mean(res2, axis=0)
        res = np.mean(res, axis=0)
        ttxt.append(txt_num)
        max_a = max(max_a, np.linalg.norm((res[1] + res[0]) / 2 - txt_num * 52.78 / 1920))
        max_a2 = max(max_a2, np.linalg.norm(res2 - txt_num * 52.78 / 1920.0))
        ave.append((res[1] + res[0]) / 2 - txt_num * 52.78 / 1920)
        left.append(res[0] - txt_num * 52.78 / 1920.0)
        right.append(res[1] - txt_num * 52.78 / 1920.0)
        ave_2.append(res2 - txt_num * 52.78 / 1920.0)
        value_np.append(np.linalg.norm(res2 - txt_num * 52.78 / 1920.0))
        t=int((txt_num[0]//640)*3 + txt_num[1]//360)
        era_value[t]+=np.linalg.norm(res2 - txt_num * 52.78 / 1920.0)
        era_times[t]+=1.
        dis_left.append(res[0])
        dis_right.append(res[1])
        img_ps.append(img_p)
        g_len = np.fabs((res[0][0] + res[1][0]) / 2)
        r_truth.append((res[0] * (52.78 - g_len) / 52.78) + (res[1] * g_len / 52.78) - txt_num * 52.78 / 1920.0)
        imgs.append(image)
        vec_left += (res[0] - txt_num * 52.78 / 1920.0)
        vec_right += (res[1] - txt_num * 52.78 / 1920.0)
        vec_ave += ((res[1] + res[0]) / 2 - txt_num * 52.78 / 1920)
        # print(res2,txt_num)
        vec_ave_2 += (res2 - txt_num * 52.78 / 1920.0)
        tt += 1

    vec_left /= tt
    vec_right /= tt
    vec_ave /= tt
    vec_ave_2 /= tt
    user32 = ctypes.windll.user32
    '''
    dg = data_geter('')
    user32.SetProcessDPIAware(2)
    screen = pygame.display.set_mode((user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)))
    background_color = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
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
        c = int(np.linalg.norm(a) * 255.0 / max_a)
        color = (c, 0, 0)
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
        c = int(np.linalg.norm(a) * 255.0 / max_a)
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
    '''
    print(f'cali {time2 - time1}')
    print(f'{timeit.default_timer() - time1}')
    print(f'{read_img} {read_img / img_times}')
    print(f'{read_txt} {read_txt/txt_times}')
    gt=et.get_time()
    t=-4
    for g in gt:
        if isinstance(g,list):
            # print('tuple')
            for i in g:
                print(f'{t} {i}')
                t+=1
        else:
            print(g)
    # print(f'-----------------------------max {max_a} {max_a2}')
    # print('{:<5}{:<30}\t{:<30}\t{:<30}\t{:<30}'.format('', '左眼误差', '右眼误差', '两眼平均误差', '平均修正误差'))
    # # print(('左眼误差右眼误差两眼平均误差 平均修正误差'))
    # for i, a, b, c, d, e in zip(range(len(left)), left, right, ave, ave_2, img_ps):
    #     print('{:<3}: {:<30}\t{:<30}\t{:<30}\t{:<30}\t{}\t'.format(i, str(a), str(b), str(c), str(d), e))
    # print('{:<30}{:<30}{:<30}{:<30}'.format('左眼误差均值', '右眼误差均值', '两眼平均误差均值', '平均修正误差均值'))
    # print('{:<30}\t{:<30}\t{:<30}{:<30}'.format(str(vec_left), str(vec_right), str(vec_ave), str(vec_ave_2)))
    print(f'nine areas {era_value/era_times}')


def cali():
    global read_img, img_times, txt_times, read_txt
    tracker1 = et.eye_tracker()
    tracker1.set_calibration([np.array([[0., 0.] for i in range(7)]), np.array([[0., 0.] for i in range(7)])],
                             [np.array([[0., 0.] for i in range(7)]), np.array([[0., 0.] for i in range(7)])])
    vecs_left = []
    vecs_right = []
    vecs_ave = []
    c_txt = []
    count_t=0
    des_left, des_right, des_ave = [], [], []
    txtss, imgss, imagess = get_path('cali')
    for txts, imgs, images in zip(txtss, imgss, imagess):
        ress = []
        count_k=0
        txt_num = np.zeros((2))
        for txt, img_p, image in zip(txts, imgs, images):
            # print(img_p)
            time1=timeit.default_timer()
            img = cv2.imread(img_p)
            time2 = timeit.default_timer()
            read_img+=time2-time1
            img_times+=1.
            gray_img = du.get_gray_pic(img)
            res = tracker1.detect(gray_img, image)
            if isinstance(res, str):
                # print(f'{res} pass')
                continue
            time3=timeit.default_timer()
            with open(txt) as t:
                txt = t.read()
            txt = txt.split(' ')
            txt_num = np.array(txt, dtype=np.float32)
            time4=timeit.default_timer()
            txt_times+=1.
            read_txt+=time4-time3
            ress.append(res)
            count_k+=1
        count_t += 1
        average = np.mean(ress, axis=0)
        if len(ress) == 0:
            continue
        res = []
        for r1 in ress:
            if not np.linalg.norm(r1 - average) > 10:
                res.append(r1)
        if len(res) == 0:
            continue
        print(count_t)
        res = np.mean(res, axis=0)
        # print(f't{count_t} k{count_k}')
        c_txt.append(txt_num)
        g_len = np.fabs((res[0][0] + res[1][0]) / 2)
        vecs_right.append((res[1] - txt_num * 52.78 / 1920))
        vecs_left.append((res[0] - txt_num * 52.78 / 1920))
        vecs_ave.append((res[0] * (52.78 - g_len) / 52.78) + (res[1] * (g_len) / 52.78) - txt_num * 52.78 / 1920)
        des_left.append(res[0])
        des_right.append(res[1])
        des_ave.append((res[0] * (52.78 - g_len) / 52.78) + (res[1] * (g_len) / 52.78))

    return np.array(vecs_left, dtype=np.float32), \
           np.array(vecs_right, dtype=np.float32), \
           np.array(vecs_ave, dtype=np.float32), \
           np.array(des_left, dtype=np.float32), np.array(des_right, dtype=np.float32), np.array(des_ave,
                                                                                                 dtype=np.float32)

    # return np.array(vecs_left, dtype=np.float32), \
    #        np.array(vecs_right, dtype=np.float32), \
    #        np.array(vecs_ave, dtype=np.float32), \
    #        np.array(c_txt, dtype=np.float32)


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
    calis=tracker_1.calibration(cap=cap, screen=screen, root_path='C:\\Users\\snapping\\Desktop\\data\\2022.11.12\\wtc',
                          cali_nums=20, store=True, points=points)
    # calis=[]
    # for i in range(7):
    #     calis.append(np.array([[0.,0.] for i in range(7)],dtype=np.float32))
    # calis=tuple(calis)
    tracker_1.tracking(cap,*calis)


if __name__ == '__main__':
    test()
    # cali()
    # main()