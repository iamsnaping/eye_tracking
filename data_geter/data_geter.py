import pygame
import time
import ctypes
from eye_utils import data_util as du
import cv2
import os
import random


class data_geter:
    def __init__(self, root_path):
        user32 = ctypes.windll.user32

        user32.SetProcessDPIAware(2)
        self.root_path = root_path
        self.cali_path = os.path.join(self.root_path, 'cali')
        self.test_path = os.path.join(self.root_path, 'test')
        self.con_path = os.path.join(self.root_path, 'com')
        self.pic_path = self.test_path
        self.txt_path = self.test_path
        if not os.path.exists(self.cali_path):
            os.makedirs(self.cali_path)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        if not os.path.exists(self.con_path):
            os.makedirs(self.con_path)
        self.screenwidth = user32.GetSystemMetrics(0)
        self.screenheight = user32.GetSystemMetrics(1)
        self.grid_num = 5

    def get_data(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, self.screenwidth)
        cap.set(4, self.screenheight)
        pygame.init()
        screen = pygame.display.set_mode((self.screenwidth, self.screenheight))
        pygame.display.set_caption("data geter")
        points = self.get_cali_points()
        # self.get_com(cap,screen,points,[11,11])
        # breakpoint()
        self.pic_path=self.cali_path
        self.txt_path=self.cali_path
        self.get_pic(cap, screen, points)
        points = self.get_points(20, self.screenwidth, self.screenheight)
        self.pic_path=self.test_path
        self.txt_path=self.test_path
        self.get_pic(cap, screen, points)
    def get_com(self,cap,screen,points,nums):
        self.pic_path=self.con_path
        self.txt_path=self.con_path
        p=[]
        for i in nums:
            p.append(points[i])
        self.get_pic(cap,screen,p)

    def show_cali(self):
        background_color = (255, 255, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        pygame.init()
        screen = pygame.display.set_mode((self.screenwidth, self.screenheight))
        pygame.display.set_caption("data geter")
        points = self.get_cali_points()
        screen.fill(background_color)
        for point in points:
            pygame.draw.circle(screen, RED, (point[0], point[1]), 10)
            pygame.display.update()
        time.sleep(10)


    def get_cali_points(self):
        x=200
        y=100
        stride_x = (self.screenwidth) / self.grid_num
        stride_y = (self.screenheight-200) / self.grid_num
        points = []
        # for i in range(4):
        #     for j in range(4):
        #         points.append([i*stride_x+x,j*stride_y+y])
        # return points
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                if i == 0 or j == 0:
                    continue
                points.append([i * stride_x, j * stride_y])
        return points

    def get_points(self, num, width=1920, height=1080):
        x=width/16
        y=height/9
        o_x=x/2
        o_y=y/2
        print(8*y+o_x)
        print(15*x+o_x)
        points=[]
        for i in range(16):
            for j in range(9):
                points.append([x*i+o_x,j*y+o_y])
        return points


    def get_pic(self, cap, screen, points):
        background_color = (255, 255, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        t = 0
        for point in points:
            screen.fill(background_color)
            pygame.draw.circle(screen, RED, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            pygame.draw.circle(screen, BLUE, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            for i in range(4):
                sucess, img = cap.read()
            cv2.imwrite(os.path.join(self.pic_path, str(t) + '.png'), img)
            p = str(point[0]) + ' ' + str(point[1])
            file_path = os.path.join(self.txt_path, str(t) + '.txt')
            with open(file_path, 'w') as fd:
                fd.write(str(p))
                fd.close()
            pygame.draw.circle(screen, GREEN, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            t += 1


if __name__ == '__main__':
    dg = data_geter('C:\\Users\\snapping\\Desktop\\data\\2022.11.1\\wtc')
    dg.get_data()
    dg.show_cali()
