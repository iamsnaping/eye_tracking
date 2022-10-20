#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :opfacevideo.py
# @Time      :2020/6/20 16:23
# @Author    :Raink
import time

import threading
import cv2
import os
# import openface

# fileDir = os.path.dirname(os.path.realpath(__file__))
# modelDir = os.path.join(fileDir, 'models')
# dlibModelDir = os.path.join(modelDir, 'dlib')
#
# align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))


class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name
        self.times=0
        # self.frame = np.zeros([400, 400, 3], np.uint8)
#480 640
    def run(self):
        capture = cv2.VideoCapture(self.cam_name)
        root_path='C:\\Users\\snapping\\Desktop\\eye_tracking\\gray_img'
        while (True):
            # 获取一帧
            ret, frame = capture.read()
            # 获取的帧送入检测，绘制检测结果后返回,自拍模式做镜像
            show_img,flag= self._detector(frame, mirror=True)
            if not isinstance(flag,bool):
                print(flag)
                file_name=time.strftime('%y%m%d%I%M%S',time.localtime())+'.png'
                file_name=os.path.join(root_path,file_name)
                self.times+=1
                # show_img=show_img[flag[1][0]:flag[1][1],flag[0][0]:flag[0][1]]
                trans_img = show_img[flag[1][0]:flag[1][1],:]
                gray_img=cv2.cvtColor(trans_img,cv2.COLOR_BGR2GRAY)
                print(gray_img.shape)
                cv2.imwrite(file_name,gray_img)

            cv2.imshow(self.win_name, show_img)
            cv2.waitKey(1)

    def _detector(self, frame, mirror=False):
        show_img = cv2.flip(frame, flipCode=1) if mirror else frame
        rects = align.getAllFaceBoundingBoxes(show_img)
        flag=True
        if len(rects) > 0:
            #i[x,y]->[y][x]
            bb = align.findLandmarks(show_img, rects[0])
            m_x=[1000,-10]
            m_y=[1000,-10]
            print(bb[36:46],end=' ')
            y=0
            x=0
            for i in bb[36:46]:
                if i[1]>480:
                   y=480
                elif i[1]<0:
                    y=0
                else:
                    y=i[1]
                if i[0]>640:
                    x=640
                elif i[0]<0:
                    x=0
                else:
                    x=i[0]
                m_x[0],m_x[1]=min(m_x[0],x),max(m_x[1],x)
                m_y[0],m_y[1]=min(m_y[0],y),max(m_y[1],y)
            m_y[0]=max(0,m_y[0]-50)
            m_y[1]=min(480,m_y[1]+50)
                # print(f'face 0 {bb[0]}')
            flag=(m_x,m_y)
            # for pt in bb:
            #     cv2.circle(show_img, pt, 3, [0, 0, 255], thickness=-1)
            # cv2.circle(show_img, bb[0], 3, [0, 0, 255], thickness=-1)

        return show_img,flag


if __name__ == "__main__":
    camera1 = OpcvCapture("camera1", 0)
    camera1.start()