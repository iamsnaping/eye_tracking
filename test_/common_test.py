#!/usr/bin/env python
import functools
import itertools
import math
import os
import time
import cv2, queue, threading, time
from collections import deque

import numpy as np
import pynput.keyboard

from base_estimation.plcr import plcr
import cv2
import pygame
# es_module=plcr.plcr()
import time,timeit

import pyautogui
from pynput import keyboard

# def cmp(a,b):
#     if np.abs(a[1]-b[1])<2:
#         if a[0]<b[0]:
#             return -1
#         return 1
#     if a[1]<b[1]:
#         return -1
#     return 1
#
# a=[np.array([52.78,31.26]),np.array([52.78,0.]),np.array([52.78/2,32.]),np.array([0,31.26]),np.array([0,0])]
# a.sort(key=functools.cmp_to_key((cmp)))
# b=a
# matrix=[]
# print(a)
# for i in itertools.combinations(a,3):
#     print(i)
#
# for point in b:
#     l=[]
#     for p in b:
#        l.append(np.linalg.norm(point-p))
#     l=np.array(l)
#     l/=l.sum()
#     matrix.append(l)
# for m in matrix:
#     print(m.tolist(),m@m)
#
# print(matrix[0]@matrix[1])


def on_press(key):
    if isinstance(key,pynput.keyboard.KeyCode):
        print(type(key.char))
        if key.char =='a':
            print('123123123')
            return False
    print(key)

def run_key():
    with keyboard.Listener(on_press=on_press) as lsn:
        lsn.join()


t=threading.Thread(target=run_key)
t.start()
print('123')