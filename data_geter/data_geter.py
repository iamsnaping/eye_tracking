import PyHook3 as pyHook
import pythoncom  # 没这个库的直接pip install pywin32安装
import pyautogui
import os
import cv2
from eye_utils import data_util as du


root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.22\\'


def funcMiddle(event):
    if (event.MessageName != "mouse move"):  # 因为鼠标一动就会有很多mouse move，所以把这个过滤下
        global j
        j = j + 1
        # print('第{:3d}次：按下鼠标中键我就会出现，嘻嘻'.format(j),end=' ')
    return pyautogui.position()


def main():
    cap=cv2.VideoCapture(0)
    cap.set(3,1920)
    cap.set(4,1080)
    # cap.set(10,200)

    # 关闭白平衡，解决偏色问题
    # print(cap.set(cv2.CAP_PROP_AUTO_WB, 0))
    # 设置曝光为手动模式
    # print(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1))
    # 设置曝光的值为0
    # print(cap.set(cv2.CAP_PROP_EXPOSURE, 100))
    t=0
    print('prepared')
    while True:
        sucess,img=cap.read()
        du.show_ph(img,wait_time=5000)
        sucess,img=cap.read()
        pic_path=os.path.join(root_path,str(t)+'.png')
        file_path=os.path.join(root_path,str(t)+'.txt')
        cv2.imwrite(pic_path,img)
        position = pyautogui.position()
        p=str(position[0])+" "+str(position[1])
        with open(file_path,'w') as fd:
            fd.write(str(p))
        t+=1

if __name__=='__main__':
    main()