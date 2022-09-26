import cv2
from heapq import *
import numpy as np
import timeit
import matplotlib.pylab as plt
import numpy as np
import pylab
from scipy import signal
import scipy.fftpack as fp
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

path='C:\\Users\\22848\\Desktop\\pic_detect\\9.24\\img_test4.png'

def show_ph(img,name='img'):
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name,1920,1080)
    cv2.imshow(name,img)
def get_ph():
    img= cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def get_round(pa):
    img=pa.copy()



def get_blog(pa):
    img=pa.copy()
    mean_p=img.mean()
    width=img.shape[0]
    length=img.shape[1]
    params = cv2.SimpleBlobDetector_Params()
    for i in range(width):
        for j in range(length):
            img[i][j]=max(0,int(img[i][j])-int(mean_p)-10)
    params.minThreshold = 20
    params.maxThreshold = 200
    # 255 亮 0 暗
    params.filterByColor = True
    params.blobColor = 255
    # 根据面积过滤
    # 按大小:可以根据大小过滤Blobs，方法是设置参数filterByArea = 1，以及适当的minArea和maxArea值。
    # 例如，设置minArea = 100将过滤掉所有像素个数小于100的Blobs。
    params.filterByArea = True
    params.minArea = 40
    params.maxArea = 200
    # 要根据圆度进行过滤，设置filterByCircularity = 1。然后设置适当的minCircularity和maxCircularity值。
    params.filterByCircularity = True
    # 4PIS/(C**2)
    params.minCircularity = 0.1
    params.maxCircularity=1
    # 根据Convexity过滤，这个参数是(凹凸性)
    # 凸性定义为(Blob的面积/它的凸包的面积)。现在，凸包的形状是最紧的凸形状，完全包围了形状。
    # 设置filterByConvexity = 1，然后设置0≤minConvexity≤1和maxConvexity(≤1)。
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.maxConvexity = 1

    # 根据Inertia过滤,惯性比
    # 它衡量的是一个形状的伸长程度。例如，对于圆，这个值是1，对于椭圆，它在0和1之间，对于直线，它是0。
    # 初步可以认为是外接矩形的长宽比，圆的外接矩形的长宽相等，椭圆是有长短轴，短轴长度除以长轴长度，介于0~1
    # 直线可以认为没有宽度，因此是0
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio=1

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
def test():
    img=get_ph()
    img=get_blog(img)
    show_ph(img,name='blob')
    cv2.waitKey()

def get_mid(img):
    return cv2.medianBlur(img,ksize=5)

def process_video():
    cap = cv2.VideoCapture('C:\\Users\\22848\\Desktop\\pic_detect\\2022-09-24_165258_313.avi')
    while (cap.isOpened()):
        ref, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if ref == True:
            frame=get_mid(frame)
            frame=get_blog(frame)
            show_ph(frame)
            cv2.waitKey(10)
        else:
            break

    cap.release()

if __name__=='__main__':
    process_video()