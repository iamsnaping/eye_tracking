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
        self.pic_path = self.cali_path
        self.txt_path = self.cali_path
        self.get_pic(cap, screen, points, turns=21)
        points = self.get_points(20, self.screenwidth, self.screenheight)
        time.sleep(5)
        self.pic_path = self.test_path
        self.txt_path = self.test_path
        self.get_pic(cap, screen, points)

    def get_calibration(self,cap):
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, self.screenwidth)
        cap.set(4, self.screenheight)
        pygame.init()
        screen = pygame.display.set_mode((self.screenwidth, self.screenheight))
        pygame.display.set_caption("data geter")
        points = self.get_cali_points()


    def get_com(self, cap, screen, points, nums):
        self.pic_path = self.con_path
        self.txt_path = self.con_path
        p = []
        for i in nums:
            p.append(points[i])
        self.get_pic(cap, screen, p)

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
        pygame.image.save(screen, os.path.join('C:\\Users\\snapping\\Desktop', 'calibratin.png'))
        screen.fill(background_color)
        points = self.get_points(20, self.screenwidth, self.screenheight)
        for point in points:
            pygame.draw.circle(screen, RED, (point[0], point[1]), 10)
            pygame.display.update()
        pygame.image.save(screen, os.path.join('C:\\Users\\snapping\\Desktop', 'getpoints.png'))

    def get_cali_points(self):
        x = 200
        y = 100
        points = []
        points.append([self.screenwidth / 2, self.screenheight / 2])
        points.append([300, 60])
        points.append([self.screenwidth / 2, 60])
        points.append([self.screenwidth - 300, 60])
        points.append([300, self.screenheight - 60])
        points.append([self.screenwidth / 2, self.screenheight - 60])
        points.append([self.screenwidth - 300, self.screenheight - 60])
        return points

    def get_points(self, num, width=1920, height=1080):
        x = width / 16
        y = height / 9
        o_x = x / 2
        o_y = y / 2
        points = []
        for i in range(9):
            for j in range(16):
                points.append([x * j + o_x, i * y + o_y])
        return points

    def get_pic_not_stored(self,cap,screen,points):
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
            p = os.path.join(self.pic_path, str(t))
            if not os.path.exists(p):
                os.makedirs(p)
                ref, img = cap.retrieve()
                if t < 1:
                    continue
                if not ref:
                    continue
                cv2.imwrite(os.path.join(p, str(t - 1) + '.png'), img)
                content = str(point[0]) + ' ' + str(point[1])
                file_path = os.path.join(p, str(t - 1) + '.txt')
                print(file_path)
                with open(file_path, 'w') as fd:
                    fd.write(str(content))
                    fd.close()
            pygame.draw.circle(screen, GREEN, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            t += 1
    def get_pic(self, cap, screen, points, turns=6):
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
            p = os.path.join(self.pic_path, str(t))
            if not os.path.exists(p):
                os.makedirs(p)
            for i in range(turns):
                ref, img = cap.read()
                if i < 1:
                    continue
                if not ref:
                    continue
                cv2.imwrite(os.path.join(p, str(i - 1) + '.png'), img)
                content = str(point[0]) + ' ' + str(point[1])
                file_path = os.path.join(p, str(i - 1) + '.txt')
                print(file_path)
                with open(file_path, 'w') as fd:
                    fd.write(str(content))
                    fd.close()
            pygame.draw.circle(screen, GREEN, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            t += 1


if __name__ == '__main__':
    dg = data_geter('C:\\Users\\snapping\\Desktop\\data\\2022.11.8\\wtc')
    # dg.get_data()
    dg.show_cali()
    # user32 = ctypes.windll.user32
    # user32.SetProcessDPIAware(2)
    # screen = pygame.display.set_mode((user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)))
    # background_color = (255, 255, 255)
    # RED = (0, 0, 255)
    # BLUE = (255, 0, 0)
    # GREEN = (0, 255, 0)
    # pygame.init()
    # pygame.display.set_caption("data geter")
    # t = 0
    # screen.fill((255,0,0))
    # pygame.image.save(screen, 'C:\\Users\\snapping\\Desktop\\data\\2022.11.1\\wtc\\sample1.png')
    # screen.fill((200, 0, 0))
    # pygame.image.save(screen, 'C:\\Users\\snapping\\Desktop\\data\\2022.11.1\\wtc\\sample2.png')
    # screen.fill((150, 0, 0))
    # pygame.image.save(screen, 'C:\\Users\\snapping\\Desktop\\data\\2022.11.1\\wtc\\sample3.png')
    # screen.fill((100, 0, 0))
    # pygame.image.save(screen, 'C:\\Users\\snapping\\Desktop\\data\\2022.11.1\\wtc\\sample4.png')
    # screen.fill((50, 0, 0))
    # pygame.image.save(screen, 'C:\\Users\\snapping\\Desktop\\data\\2022.11.1\\wtc\\sample5.png')
