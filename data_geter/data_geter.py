#!/usr/bin/env python
import pygame
import time
import ctypes
from eye_utils import data_util as du
import cv2
import os
import random

user32 = ctypes.windll.user32

user32.SetProcessDPIAware(2)
[screenWidth, screenHeight] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.27\\wtc1\\glasses'
calibration_path = os.path.join(root_path, 'cali')
test_path = os.path.join(root_path, 'test')
if not os.path.exists(calibration_path):
    os.makedirs(calibration_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
(width, height) = (40, 40)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, screenWidth)
cap.set(4, screenHeight)
background_color = WHITE

pygame.init()
screen = pygame.display.set_mode((1920, 1080))
pygame.display.set_caption("VPN-Status")
screen.fill(background_color)
pygame.display.update()
t = 0
nums = [[200, 360], [200, 720], [640, 880], [1280, 880], [1720, 720], [1720, 360], [1280, 200], [640, 200], [960, 540]]
# nums=[nums[1] for i in range(5)]
imgs=[]
for i in range(5):
    sucess, img = cap.read()
    cv2.imwrite(os.path.join(calibration_path, str(t) + '.png'), img)
# nums=[nums[8] for i in range(5)]
for num in nums:
    screen.fill(background_color)
    pygame.draw.circle(screen, RED, (num[0], num[1]), 10)
    pygame.display.update()
    time.sleep(2.0)
    pygame.draw.circle(screen, (0, 0, 255), (num[0], num[1]), 10)
    pygame.display.update()
    time.sleep(1.0)
    for i in range(5):
        sucess, img = cap.read()
    # pic_path=os.path.join(root_path,str(t)+'.png')

    cv2.imwrite(os.path.join(calibration_path, str(t) + '.png'), img)
    # du.show_ph(img)
    p = str(num[0]) + ' ' + str(num[1])
    file_path = os.path.join(calibration_path, str(t) + '.txt')
    with open(file_path, 'w') as fd:
        fd.write(str(p))
        fd.close()
    pygame.draw.circle(screen, (0, 255, 0), (num[0], num[1]), 10)
    pygame.display.update()
    time.sleep(2.0)
    t += 1
breakpoint()
for i in range(25):
    x=random.randint(200,1720)
    y=random.randint(200,880)
    screen.fill(background_color)
    pygame.draw.circle(screen, RED, (x, y), 10)
    pygame.display.update()
    time.sleep(2.0)
    pygame.draw.circle(screen, (255, 0, 255), (x, y), 10)
    pygame.display.update()
    time.sleep(1.0)
    sucess, img = cap.read()
    # pic_path=os.path.join(root_path,str(t)+'.png')
    cv2.imwrite(os.path.join(test_path, str(t) + '.png'), img)
    p = str(x) + ' ' + str(y)
    file_path = os.path.join(test_path, str(t) + '.txt')
    with open(file_path, 'w') as fd:
        fd.write(str(p))
        fd.close()
    t += 1
    pygame.draw.circle(screen, (0, 255, 0), (num[0], num[1]), 10)
    time.sleep(1.0)