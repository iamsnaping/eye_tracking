import time
from heapq import *
import threading
import cv2
import os
import numpy as np

class pupil_detection(object):
    def __init__(self,win_name, cam_name):
        self.cam_name = cam_name
        self.win_name = win_name
        self.times = 0

    def cap_pic(self):
        capture = cv2.VideoCapture(self.cam_name)
        root_path = 'C:\\Users\\22848\\Desktop\\pic_detect'
        t=0
        while (True):
            # 获取一帧
            ret, frame = capture.read()
            # print(frame.shape)
            cv2.imshow('cap',frame)
            # frame=self.blob(frame)
            # cv2.imshow('cap', frame)
    def pupil_detect_blob(self,pa):
        img = pa
        leng = pa.shape[0]
        wid = pa.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img, 9)
        # cv2.imshow('blur', img)
        # cv2.waitKey()
        li = []
        for i in img:
            li.extend(i.tolist())
        heapify(li)
        ll = nsmallest(2500, li)
        for i in range(leng):
            for j in range(wid):
                img[i][j] = 0 if img[i][j] <= ll[-1] else 255

        # cv2.imshow('img', img)
        # cv2.waitKey()

        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByColor = True
        params.blobColor = 0
        params.filterByArea = True
        params.minArea = 500
        params.maxArea = 4000
        params.filterByCircularity = True
        # 4PIS/(C**2)
        params.minCircularity = 0.3
        detector = cv2.SimpleBlobDetector_create(params)
        # 检测blobs
        keypoints = detector.detect(img)
        keypoints_array = np.array(keypoints)

        # 用红色圆圈画出检测到的blobs
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 确保圆的大小对应于blob的大小
        im_with_keypoints = cv2.drawKeypoints(pa, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints

    def hough(self,pa):
        img=pa
        leng = pa.shape[0]
        wid = pa.shape[1]
        img =cv2.medianBlur(img,5)
        img = cv2.Laplacian(img, -1,cv2.CV_64F,ksize=5)
        lap=cv2.Laplacian(img,-1,3)
        for i in range(leng):
            for j in range(wid):
                t=max(0,lap[i][j])
                img[i][j] = max(0, int(img[i][j])-int(t))
        cv2.imshow('lap',img)
        cv2.waitKey()
        #1. 图片 2. 方法 3.推荐1.5。4.圆心 5.这个值通常要设置得更大。6.圆度 7.最小半径 8.最大半径
        circles=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT_ALT,1.5, 100,param1=200,param2=0.4,minRadius=30,maxRadius=50)
        print(f'circles {circles}')
        circle_number = 0
        for i in circles[0, :]:
            if len(circles) > 1:
                print('不止一个圆')
            else:
                cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 255), 2)  # 画圆
                cv2.circle(img, (int(i[0]), int(i[1])), 2, (255, 0, 255), 2)  # 画圆心
                circle_number += 1
        print(f'circle num_{circle_number}')
        cv2.imshow('round',img)
        cv2.waitKey()




    def blob(self,pa):
        # img = cv2.imread(pa)[200:400][:]
        img=pa
        leng=pa.shape[0]
        wid=pa.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('origin',img)
        mid = img.mean()
        img=cv2.GaussianBlur(img,(5,5),10,10)
        # cv2.imshow('blur',img)
        for i in range(leng):
            for j in range(wid):
                img[i][j]=255 if img[i][j]>mid else 0
        # cv2.imshow('blur',img)
        # 设置SimpleBlobDetector参数
        params = cv2.SimpleBlobDetector_Params()

        # 改变阈值
        # 只会检测minThreshold 和 maxThreshold之间的
        params.minThreshold = 0
        params.maxThreshold = 255
        #255 亮 0 暗
        params.filterByColor=True
        params.blobColor=255
        # 根据面积过滤
        # 按大小:可以根据大小过滤Blobs，方法是设置参数filterByArea = 1，以及适当的minArea和maxArea值。
        # 例如，设置minArea = 100将过滤掉所有像素个数小于100的Blobs。
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 100
        # 要根据圆度进行过滤，设置filterByCircularity = 1。然后设置适当的minCircularity和maxCircularity值。
        params.filterByCircularity = True
        #4PIS/(C**2)
        params.minCircularity = 0.1
        # 根据Convexity过滤，这个参数是(凹凸性)
        # 凸性定义为(Blob的面积/它的凸包的面积)。现在，凸包的形状是最紧的凸形状，完全包围了形状。
        # 设置filterByConvexity = 1，然后设置0≤minConvexity≤1和maxConvexity(≤1)。
        # params.filterByConvexity = True
        # params.minConvexity = 0.1
        # params.maxConvexity = 1

        # 根据Inertia过滤,惯性比
        # 它衡量的是一个形状的伸长程度。例如，对于圆，这个值是1，对于椭圆，它在0和1之间，对于直线，它是0。
        # 初步可以认为是外接矩形的长宽比，圆的外接矩形的长宽相等，椭圆是有长短轴，短轴长度除以长轴长度，介于0~1
        # 直线可以认为没有宽度，因此是0
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.5

        # 创建一个带有参数的检测器
        detector = cv2.SimpleBlobDetector_create(params)
        # 检测blobs
        keypoints = detector.detect(img)

        # 用红色圆圈画出检测到的blobs
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 确保圆的大小对应于blob的大小
        im_with_keypoints = cv2.drawKeypoints(pa, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 结果显示
        return im_with_keypoints
        # cv2.imshow("blob", im_with_keypoints)
        # cv2.waitKey()


if __name__=='__main__':
    cap_pic=pupil_detection("camera1", 1)
    cap_pic.cap_pic()
    # cap_pic.hough('C:\\Users\\22848\\Desktop\\pic_detect\\1.png')
