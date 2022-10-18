import copy
import math
import cv2
from PIL import Image
from heapq import *
import numpy as np
import timeit
import matplotlib.pylab as plt
import numpy as np
import os
from eye_utils import data_util as du
import pylab
from scipy import signal
import scipy.fftpack as fp
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from skimage.feature import blob_log, blob_doh, blob_dog
from sklearn.cluster import KMeans

root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.9\\'
video_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.9\\2022-10-09_143159_334.avi'
pic_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.9\\origin'
video_test_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.9\\video_test'
T_R=0
T_G=0

def show_ph(img, name='img', wait_time=False, pass_=True):
    if pass_:
        return
    width = img.shape[1]
    length = img.shape[0]
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, width, length)
    cv2.imshow(name, img)
    if wait_time == False:
        cv2.waitKey()
    else:
        cv2.waitKey(wait_time)


def get_ph(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img=cv2.GaussianBlur(img,(3,3),1)
    return img


def cmp(x, y):
    if x[2] <= y[2]:
        return -1
    else:
        return 1


def glints_blur(img, blobs):
    kernel = np.zeros((11, 11))
    width = img.shape[0]
    length = img.shape[1]
    for i in range(11):
        for j in range(11):
            if (np.abs(i - 5) ** 2) + (np.abs(j - 5) ** 2) <= 16:
                kernel[i][j] = 1.0 / 40.0
            else:
                kernel[i][j] = -1.0 / 60.0
    blob_feature = []
    for blob in blobs:
        # if img[i][j]>220:
        y, x = int(blob.pt[0]), int(blob.pt[1])
        sub = np.zeros((11, 11), dtype=np.float64)
        wid = min(x + 6, width) - max(0, x - 5)
        leg = min(y + 6, length) - max(0, y - 5)
        if wid < 11 or leg < 11:
            continue
        sub[0:wid, 0:leg] = img[(max(0, x - 5)):(min(x + 6, width)), (max(0, y - 5)):(min(y + 6, length))]
        value = (sub * kernel).sum()
        if value < 50:
            continue
        blob_feature.append([blob, value])
    return blob_feature


def round_blur(img_copy, k_size=11, r_size=8):
    glings = []
    width = img_copy.shape[0]
    length = img_copy.shape[1]
    kernel = np.zeros((11, 11))
    for i in range(11):
        for j in range(11):
            if (np.abs(i - 5) ** 2) + (np.abs(j - 5) ** 2) <= 16:
                kernel[i][j] = 1.0 / 50.0
            else:
                kernel[i][j] = -1.0 / 70.0
                # kernel[i][j]=0
    for i in range(width):
        for j in range(length):
            if img_copy[i][j] > 220:
                sub = np.zeros((11, 11))
                wid = min(i + 6, 1080) - max(0, i - 5)
                leg = min(j + 6, 1920) - max(0, j - 5)
                if wid < 11 or leg < 11:
                    continue
                sub[0:wid, 0:leg] = img_copy[(max(0, i - 5)):(min(i + 6, 1080)), (max(0, j - 5)):(min(j + 6, 1920))]
                img_copy[i][j] = max(0, (sub * kernel).sum())
                img_copy[i][j] = min(img_copy[i][j], 255)
                img_copy[i][j] = 0 if img_copy[i][j] < 50 else 255
                if img_copy[i][j] == 255:
                    glings.append([i, j])
            else:
                img_copy[i][j] = 0
    return img_copy, glings


def get_glints(img):
    # mean_p = img.mean()
    width = img.shape[0]
    length = img.shape[1]
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 180
    params.maxThreshold = 240
    # 255 亮 0 暗
    params.filterByColor = True
    params.blobColor = 255
    params.minDistBetweenBlobs=5
    # params.minRepeatability=1
    # 根据面积过滤
    # 按大小:可以根据大小过滤Blobs，方法是设置参数filterByArea = 1，以及适当的minArea和maxArea值。
    # 例如，设置minArea = 100将过滤掉所有像素个数小于100的Blobs。
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 100
    # 要根据圆度进行过滤，设置filterByCircularity = 1。然后设置适当的minCircularity和maxCircularity值。
    # # 4PIS/(C**2)
    params.filterByCircularity = True
    params.minCircularity = 0
    params.maxCircularity = 1
    # 根据Convexity过滤，这个参数是(凹凸性)
    # 凸性定义为(Blob的面积/它的凸包的面积)。现在，凸包的形状是最紧的凸形状，完全包围了形状。
    # 设置filterByConvexity = 1，然后设置0≤minConvexity≤1和maxConvexity(≤1)。
    params.filterByConvexity = True
    params.minConvexity = 0
    params.maxConvexity = 1

    # 根据Inertia过滤,惯性比
    # 它衡量的是一个形状的伸长程度。例如，对于圆，这个值是1，对于椭圆，它在0和1之间，对于直线，它是0。
    # 初步可以认为是外接矩形的长宽比，圆的外接矩形的长宽相等，椭圆是有长短轴，短轴长度除以长轴长度，介于0~1
    # 直线可以认为没有宽度，因此是0
    params.filterByInertia = True
    params.minInertiaRatio = 0
    params.maxInertiaRatio = 1
    img_copy=img.copy()
    for i in range(width):
        for j in range(length):
            img_copy[i][j] = 0 if img_copy[i][j] < 200 else 255
    show_ph(img_copy,'img_copy')
    blobs = get_blog(img_copy, params)
    # img_c=draw_keypoints(img,blobs)
    # show_ph(img_c)
    # for blob in blobs:
    #     print(blob.pt, blob.size)
    if len(blobs) == 0:
        img, glings = round_blur(img.copy())
        x = np.array(glings)
        if len(glings) != 0:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
            return kmeans.cluster_centers_
        return False
    # breakpoint()
    a = glints_blur(img.copy(), blobs)
    a.sort(key=lambda x: x[1], reverse=True)
    # for k in a:
    #     print(k[1])
    if len(a) <= 6:
        return a
    return a[0:6]


def get_blog(img, params):
    # 创建一个带有参数的检测器
    detector = cv2.SimpleBlobDetector_create(params)
    # 检测blobs
    keypoints = detector.detect(img)
    # img=draw_keypoints(img,keypoints)
    # print(len(keypoints))

    return keypoints


def draw_keypoints(img, keypoints, color=(0, 0, 255)):
    draw_key = cv2.drawKeypoints(img, keypoints, np.array([]), color,
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return draw_key


def test(img_path):
    img = get_ph(img_path)
    img = get_blog(img)


def get_mid(img):
    return cv2.medianBlur(img, ksize=5)


def process_video():
    process_pa = os.path.join(root_path, 'process')
    test_pa = os.path.join(root_path, 'video_test')
    if not os.path.exists(process_pa):
        os.mkdir(process_pa)
    if not os.path.exists(test_pa):
        os.mkdir(test_pa)
    t = 0
    k=-1
    dir_list=os.listdir(root_path)
    for d in dir_list:
        if os.path.splitext(d)[1]!='.avi':
            continue
        f_p=os.path.join(root_path,d)
        cap=cv2.VideoCapture(f_p)
        while cap.isOpened():
            ref, frame = cap.read()
            if t==300:
                break
            if ref:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                k+=1
                o_p = os.path.join(test_pa, str(t) + '.png')
                p_p = os.path.join(process_pa, str(t) + '.png')
                cv2.imwrite(o_p, frame)
                frame = process_pic(frame)
                cv2.imwrite(p_p, frame)
                t += 1
                # show_ph(frame, wait_time=5, pass_=False)
            else:
                break

        cap.release()


def get_data(pa):
    list_dir = os.listdir(pa)
    t = 0
    k = -1
    for v_list in list_dir:
        if os.path.splitext(v_list)[1] != '.avi':
            continue
        v_path = os.path.join(pa, v_list)
        cap = cv2.VideoCapture(v_path)
        while (cap.isOpened()):
            ref, frame = cap.read()
            k += 1
            if k % 10 != 0:
                continue
            if ref:
                img_path = os.path.join(pic_path, str(t) + '.png')
                cv2.imwrite(img_path, frame)
                t += 1
            else:
                break


def get_data_spread():
    t = 0
    k = -1
    v_list=du.get_video(root_path)
    left_p=os.path.join(root_path,'left')
    right_p=os.path.join(root_path,'right')
    if not os.path.exists(left_p):
        os.mkdir(left_p)
    if not os.path.exists(right_p):
        os.mkdir(right_p)
    for v_path in v_list:
        cap = cv2.VideoCapture(v_path)
        while (cap.isOpened()):
            ref, frame = cap.read()
            if ref:
                t += 1
                left_path=os.path.join(left_p,str(t)+'.png')
                right_path=os.path.join(right_p,str(t)+'.png')
                get_spread(frame,left_path,right_path)
            else:
                break
        cap.release()


def get_spread(img,left_path,right_path):
    img_c = du.get_gray_pic(img)
    glints = get_glints(img_c.copy())
    left = np.array([0.0, 0.0])
    right = np.array([0.0, 0.0])
    l, r = 0.0, 0.0
    # print(glints)
    if isinstance(glints, np.ndarray):
        coordinate = glints
    elif isinstance(glints, list) and len(glints) > 0:
        x, y = glints[0][0].pt[0], glints[0][0].pt[1]
        for glint in glints:
            dis = ((glint[0].pt[0] - x) ** 2) + ((glint[0].pt[1] - y) ** 2)
            if dis < 1000:
                left[0] += glint[0].pt[1]
                left[1] += glint[0].pt[0]
                l += 1
            else:
                right[0] += glint[0].pt[1]
                right[1] += glint[0].pt[0]
                r += 1
        if l != 0:
            left /= l
        if r != 0:
            right /= r
        # print(left, right)
        if left[0]<right[0]:
            coordinate = [left, right]
        else:
            coordinate=[right,left]
    else:
        return

    x, y = int(coordinate[0][0]), int(coordinate[0][1])
    x1, y1 = int(coordinate[1][0]), int(coordinate[1][1])
    width, length = img.shape[0],img.shape[1]
    a = max(0, x - 100)
    b = min(x + 100, width)
    c = max(0, y - 100)
    d = min(length, y + 100)
    a1 = max(0, x1 - 100)
    b1 = min(x1 + 100, width)
    c1 = max(0, y1 - 100)
    d1 = min(length, y1 + 100)
    # print(f'b-a d-c {b-a,d-c}')
    if b-a<200 or d-c <200 or b1-a1<200 or d1-c1<200:
        return
    eye_left = img[a:b, c:d,:]
    eye_right = img[a1:b1, c1:d1:]
    cv2.imwrite(left_path,eye_left)
    cv2.imwrite(right_path,eye_right)
    cv2.findContours()
    cv2.Canny()




def sin(x):
    return math.sin(math.pi * x / 180.0)


def cos(x):
    return math.cos(x * math.pi / 180.0)


def _():
    print(os.path.exists('C:\\Users\\snapping\\Desktop\\data\\process\\0.png'))


def get_round(img, coord=None):
    x, y = int(coord[0][0]), int(coord[0][1])
    x1, y1 = int(coord[1][0]), int(coord[1][1])
    width, length = img.shape
    a = max(0, x - 100)
    b = min(x + 100, width)
    c = max(0, y - 100)
    d = min(length, y + 100)
    a1 = max(0, x1 - 100)
    b1 = min(x1 + 100, width)
    c1 = max(0, y1 - 100)
    d1 = min(length, y1 + 100)
    eye_left = img[a:b, c:d].copy()
    eye_right = img[a1:b1, c1:d1].copy()
    show_ph(eye_left, 'left')
    show_ph(eye_right, 'right')
    width, length = eye_left.shape
    width1, length1 = eye_right.shape
    # print(width,length)
    # ksize = r * 7
    # ksize1 = r1 * 7
    # min_x = 0x3f3f3f
    # min_x1 = 0x3f3f3f
    # for i in range(0, width - ksize):
    #     for j in range(0, length - ksize):
    #         min_x = min(min_x, eye_left[i:i + ksize, j:j + ksize].mean())
    # for i in range(0, width - ksize):
    #     for j in range(0, length - ksize):
    #         min_x1 = min(min_x1, eye_right[i:i + ksize1, j:j + ksize1].mean())
    # eye_right = cv2.GaussianBlur(eye_right, (5, 5), 10, 10)
    # eye_left = cv2.GaussianBlur(eye_left, (5, 5), 10, 10)
    # eye_left=cv2.Laplacian(eye_left,-1,cv2.CV_64F,ksize=5)
    # eye_right=cv2.Laplacian(eye_right,-1,cv2.CV_64F,ksize=5)
    # for i in range(width):
    #     for j in range(length):
    #         eye_left[i][j] = 0 if (eye_left[i][j] <= 70 or eye_left[i][j] >= 220) else 255
    # for i in range(width1):
    #     for j in range(length1):
    #         eye_right[i][j] = 0 if (eye_right[i][j] <= 70 or eye_right[i][j] >= 220) else 255
    # img[:][:] = 255
    # print(f'a b c d {a,b,c,d}')
    # print(eye_left.shape)
    # print(eye_right.shape)
    # for i in range(a, b):
    #     for j in range(c, d):
    #         img[i][j] = eye_left[i - a][j - c]
    # for i in range(a1, b1):
    #     for j in range(c1, d1):
    #         img[i][j] = eye_right[i - a1][j - c1]
    # show_ph(img)
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 70
    # 255 亮 0 暗
    params.filterByColor = True
    params.blobColor = 0
    # 根据面积过滤
    # 按大小:可以根据大小过滤Blobs，方法是设置参数filterByArea = 1，以及适当的minArea和maxArea值。
    # 例如，设置minArea = 100将过滤掉所有像素个数小于100的Blobs。
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 2000
    # params.minDistBetweenBlobs=3000
    # 要根据圆度进行过滤，设置filterByCircularity = 1。然后设置适当的minCircularity和maxCircularity值。
    params.filterByCircularity = True
    # 4PIS/(C**2)
    params.minCircularity = 0.1
    params.maxCircularity = 1
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
    params.maxInertiaRatio = 1
    left= get_blog(eye_left, params)
    right=get_blog(eye_right,params)
    flag=0
    if len(left)!=0:
        flag+=1
        pt1=(left[0].pt[0]+c,left[0].pt[1]+a)
        left[0].pt=pt1
    if len(right)!=0:
        flag+=2
        pt2=(right[0].pt[0]+c1,right[0].pt[1]+a1)
        right[0].pt=pt2
    # print(left[0].pt[0],left[0].pt[1],left[0].size)
    # print(right[0].pt,right[0].size)
    return left+right
    # return get_blog(img,params)

def process_pic(img):
    img_c = img.copy()
    glints = get_glints(img_c.copy())
    left = np.array([0.0, 0.0])
    right = np.array([0.0, 0.0])
    l, r = 0.0, 0.0
    circles = []
    d_glints = []
    # print(glints)
    if isinstance(glints, np.ndarray):
        coordinate = glints
    elif isinstance(glints, list) and len(glints)>0:
        x, y = glints[0][0].pt[0], glints[0][0].pt[1]
        for glint in glints:
            dis = ((glint[0].pt[0] - x) ** 2) + ((glint[0].pt[1] - y) ** 2)
            # print(f'dis {dis}', end=' ')
            # print(glint[0].pt[1], glint[0].pt[0])
            if dis < 1000:
                left[0] += glint[0].pt[1]
                left[1] += glint[0].pt[0]
                l += 1
            else:
                right[0] += glint[0].pt[1]
                right[1] += glint[0].pt[0]
                r += 1
        # print(left, right)
        # print(l, r)
        if l != 0:
            left /= l
        if r != 0:
            right /= r
        # print(left, right)
        coordinate = [left, right]
        # print(f'l r {l,r}')
        # print(f'coordinate{coordinate}')
        for glint in glints:
            d_glints.append(glint[0])
    # print(coordinate)
    if l != 0 and r != 0:
        circles = get_round(img_c, coordinate)
        # for circle in circles:
        #     print(f'circle{circle.pt,circle.size}')

    # x, y = int(glints[0].pt[0]), int(glints[0].pt[1])
    #
    # r = int(glints[0].size*0.6)
    # x1, x2, y1, y2 = y - r - 2, y + r + 2, x - r - 2, x + r + 2
    # lu, ru, ld, rd = img_c[x1][y1], img_c[x1][y2], img_c[x2][y1], img_c[x2][y2]
    # for i in range(y - r, y + r + 1):
    #     for j in range(x - r, x + r + 1):
    #         # img_c[i][j] = (lu * (x2 - i) * (y2 - j) + ru * (i - x1) * (y2 - j) + ld * (j - y1) * (x2 - i) + rd * (
    #         #             i - x1) * (j - y1)) / ((x2 - x1) * (y2 - y1))
    #         img_c[i][j]=0
    # x, y = int(glints[1].pt[0]), int(glints[1].pt[1])
    # r = int(glints[1].size * 0.6)
    # lu, ru, ld, rd = img_c[x1][y1], img_c[x1][y2], img_c[x2][y1], img_c[x2][y2]
    # for i in range(y - r, y + r + 1):
    #     for j in range(x - r, x + r + 1):
    #         # img_c[i][j] = (lu * (x2 - i) * (y2 - j) + ru * (i - x1) * (y2 - j) + ld * (j - y1) * (x2 - i) + rd * (
    #         #         i - x1) * (j - y1)) / ((x2 - x1) * (y2 - y1))
    #         img_c[i][j]=0
    global T_R,T_G
    T_R+=(2-len(circles))
    T_G+=(6-len(d_glints))
    if len(circles) != 0:
        img = draw_keypoints(img, circles)
    if not isinstance(glints, np.ndarray):
        img = draw_keypoints(img, d_glints)
    return img


def test_round(img):
    width, length = img.shape
    for i in range(width):
        for j in range(length):
            img[i][j] = np.uint8(255) - img[i][j]
    blobs = blob_log(img, max_sigma=30, num_sigma=310, threshold=.1)


def blob_log_test(img):
    res = blob_log(img, min_sigma=10, max_sigma=20)
    # print(res)
    for blob in res:
        y, x, r = blob
        cv2.circle(img, (int(x), int(y)), int(r), thickness=2, color=(0, 0, 255))
    show_ph(img)


def test_pic(img_path):
    img = get_ph(img_path)
    img = process_pic(img)
    show_ph(img)


if __name__ == '__main__':
    # test_path = os.path.join(root_path, 'video_test')
    # img_path = os.path.join(test_path, '18.png')
    # test_pic(img_path)
    get_data_spread()
    # process_video()
    # print(f'round{T_R}')
    # print(f'glint{T_G}')
    # get_data(root_path)

    # blob_log_test(img)
    # img=get_ph(img_path)
    #
    # # test(img_path)
    # img=get_ph(img_path)
    # width,length=img.shape
    # for i in range(width):
    #     for j in range(length):
    #         img[i][j]=255 if img[i][j]>=240 else 0
    # show_ph(img)
    # _()
