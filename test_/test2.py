import cv2
import numpy as np
from pupil_detect.pupil_detect import *
cap = cv2.VideoCapture('C:\\Users\\22848\\Desktop\\pic_detect\\2022-09-24_133057_121.avi')
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# sz=(960,540)
# videoWriter = cv2.VideoWriter('C:\\Users\\22848\\Desktop\\pic_detect\\ys2.mp4',
#                                  cv2.VideoWriter_fourcc('X','V','I','D'), fps, sz)
# while True:
#     success, frame = cap.read()
#     if success:
#         img = cv2.resize(frame, sz)
#         videoWriter.write(img)
#     else:
#         print('break')
#         break
#
#     # 释放对象，不然可能无法在外部打开
# videoWriter.release()


# 2.判断是否读取成功
# pd= pupil_detection("camera1", 1)
t=0
root_p='C:\\Users\\22848\\Desktop\\pic_detect\\9.24'
while(cap.isOpened()):
    ref,frame=cap.read()
    name='.png'
    if ref==True:
        # frame=pd.hough(frame)
        cv2.namedWindow("enhanced", 0)
        cv2.resizeWindow("enhanced", 1920, 1080)
        cv2.imshow('enhanced',frame)
        key=cv2.waitKey(3)

        if key==ord('q'):
            break
        elif key==ord('w'):
            img_p = 'img_test' + str(t) + name
            t += 1
            img_pa = os.path.join(root_p, img_p)
            cv2.imwrite(img_pa,frame)
    else:
        break
cap.release()
