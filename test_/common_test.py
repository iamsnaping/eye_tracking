import time, threading
from datetime import datetime

import pynput.keyboard
from PIL import ImageGrab
# from cv2 import *
import cv2
from sklearn.datasets import make_friedman2
import numpy as np
from pynput import keyboard
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (DotProduct, WhiteKernel, RBF, Matern, PairwiseKernel, ConstantKernel,Kernel,StationaryKernelMixin,Hyperparameter,NormalizedKernelMixin,
_check_length_scale)

from abc import ABCMeta, abstractmethod
from collections import namedtuple
import math
from inspect import signature

import numpy as np
from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform


import warnings


# def video_record():   # 录入视频
#   global name
#   fps=15
#   name = datetime.now().strftime('%Y-%m-%d %H-%M-%S') # 当前的时间（当文件名）
#   screen = ImageGrab.grab() # 获取当前屏幕
#   width, high = screen.size # 获取当前屏幕的大小
#   fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D') # MPEG-4编码,文件后缀可为.avi .asf .mov等
#   video = cv2.VideoWriter('%s.avi' % name, fourcc, fps, (width, high)) # （文件名，编码器，帧率，视频宽高）
#   #print('3秒后开始录制----')  # 可选
#   #time.sleep(3)
#   print('开始录制!')
#   global start_time
#   start_time = time.time()
#   while True:
#     if flag:
#       print("录制结束！")
#       global final_time
#       final_time = time.time()
#       video.release() #释放
#       break
#     im = ImageGrab.grab()  # 屏幕抓图，图片为RGB模式
#     frame = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR) # 转为opencv的BGR模式
#     video.write(frame)  #写入vedio定义的文件
#     # time.sleep(5) # 等待5秒再次循环,控制帧数
#     #等0.1毫秒---十分之一秒--10帧/秒
# def on_press(key):   # 监听按键-home
#   global flag
#   if isinstance(key,pynput.keyboard.KeyCode):
#     if key.char=='q':
#       flag = True # 改变
#       return False # 返回False，键盘监听结束！
# def video_info():   # 视频信息
#   video = cv2.VideoCapture('%s.avi' % name)  # 记得文件名加格式不要错！
#   fps = video.get(cv2.CAP_PROP_FPS)
#   Count = video.get(cv2.CAP_PROP_FRAME_COUNT)
#   size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#   print('帧率=%.1f'%fps)
#   print('帧数=%.1f'%Count)
#   print('分辨率',size)
#   print('视频时间=%.3f秒'%(int(Count)/fps))
#   print('录制时间=%.3f秒'%(final_time-start_time))
#   print('推荐帧率=%.2f'%(fps*((int(Count)/fps)/(final_time-start_time))))
# if __name__ == '__main__':
#   flag = False
#   th = threading.Thread(target=video_record)
#   th.start()
#   with keyboard.Listener(on_press=on_press) as listener:
#     listener.join()
#   time.sleep(1)  # 等待视频释放过后
#   video_info()


'''
[[  30.842041   -93.53592  ]
 [ -59.203667  -253.09457  ]
 [ 104.91547   -306.83664  ]
 [ 274.39114   -244.08374  ]
 [ -99.604576    -9.471761 ]
 [ 108.44921    -18.593952 ]
 [ 309.9113      -1.0388947]] [[-218.45654 -200.46231]
 [-194.67258 -338.21893]
 [ -45.2334  -388.28235]
 [ 125.35728 -353.2948 ]
 [-269.46082  -98.60218]
 [ -62.67039 -146.65533]
 [ 149.0289  -121.8842 ]] [[ -81.627075  -141.7749   ]
 [ -72.11219   -260.83536  ]
 [  27.502901  -348.8265   ]
 [ 133.46564   -347.46954  ]
 [-110.62503    -14.6206455]
 [  20.669743   -84.295876 ]
 [ 155.29077   -117.34817  ]] [[ 990.84204  446.46408]
 [ 240.79633 -193.09457]
 [1064.9154  -246.83665]
 [1894.3911  -184.08374]
 [ 200.39542 1010.52826]
 [1068.4492  1001.40607]
 [1929.9113  1018.9611 ]] [[ 741.54346   339.5377  ]
 [ 105.32742  -278.21893 ]
 [ 914.7666   -328.28235 ]
 [1745.3573   -293.2948  ]
 [  30.539194  921.3978  ]
 [ 897.3296    873.34467 ]
 [1769.0289    898.1158  ]] [[ 878.3729   398.2251 ]
 [ 245.98785 -192.95406]
 [ 987.58057 -281.63223]
 [1744.7167  -280.6018 ]
 [ 203.845   1000.0473 ]
 [ 984.7037   931.256  ]
 [1766.8062   896.00085]]
'''

a = [[0, 0], [52.78, 0], [0, 31.26], [52.78, 31.26], [52.78 / 2, 31.26 + 3]]

vecs = np.array([[30.842041, -93.53592],
                 [-59.203667, -253.09457],
                 [104.91547, -306.83664],
                 [274.39114, -244.08374],
                 [-99.604576, -9.471761],
                 [108.44921, -18.593952],
                 [309.9113, -1.0388947], ], dtype=np.float32)

vecs1 = np.array([[-218.45654, -200.46231],
                  [-194.67258, -338.21893],
                  [-45.2334, -388.28235],
                  [125.35728, -353.2948],
                  [-269.46082, -98.60218],
                  [-62.67039, -146.65533],
                  [149.0289, -121.8842]], dtype=np.float32)
vecs2 = np.array([[-81.627075, -141.7749],
                  [-72.11219, -260.83536],
                  [27.502901, -348.8265],
                  [133.46564, -347.46954],
                  [-110.62503, -14.6206455],
                  [20.669743, -84.295876],
                  [155.29077, -117.34817]], dtype=np.float32)

des = np.array([[990.84204, 446.46408],
                [240.79633, -193.09457],
                [1064.9154, -246.83665],
                [1894.3911, -184.08374],
                [200.39542, 1010.52826],
                [1068.4492, 1001.40607],
                [1929.9113, 1018.9611]], dtype=np.float32)

des1 = np.array([[741.54346, 339.5377],
                 [105.32742, -278.21893],
                 [914.7666, -328.28235],
                 [1745.3573, -293.2948],
                 [30.539194, 921.3978],
                 [897.3296, 873.34467],
                 [1769.0289, 898.1158]], dtype=np.float32)
des2 = np.array([[878.3729, 398.2251],
                 [245.98785, -192.95406],
                 [987.58057, -281.63223],
                 [1744.7167, -280.6018],
                 [203.845, 1000.0473],
                 [984.7037, 931.256],
                 [1766.8062, 896.00085]], dtype=np.float32)

'''
0.9710216957018967
0.9743374337955414
0.9747209439744833
0.9755300546399969
0.9765507039039967
0.9765902998927403
0.9756408772984022
0.9999999999999863
0.9999999999999983
0.9999999999999999
'''

class RBFS(StationaryKernelMixin, NormalizedKernelMixin, Kernel):

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X / length_scale, metric="sqeuclidean")
            dists = np.sqrt(dists)
            K = np.exp(-0.5 *dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale, metric="sqeuclidean")
            dists=np.sqrt(dists)
            K = np.exp(-0.5 * (dists))

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                K_gradient = np.sqrt(K_gradient)
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                    length_scale ** 2
                )
                K_gradient=np.sqrt(K_gradient)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K
'''
0.9788436575987212
'''

a=[[387.37885 ,-203.92297]]
kernel = ConstantKernel(0.5) * RBF(length_scale=10) + WhiteKernel() * ConstantKernel(0.5)

t=0
for i in range(1,100):
    kernel = ConstantKernel(1.) *DotProduct(sigma_0=float(i)/100)+ConstantKernel(0.9) *WhiteKernel()
    g = GaussianProcessRegressor(kernel=kernel).fit(des, vecs)
    s=g.score(des, vecs)
    if t<s:
        t=s
        # print(s)
        gpr=g
kernel1 = ConstantKernel(1.65) * RBF(length_scale=7) + WhiteKernel() * ConstantKernel(0.6)
gpr2 = GaussianProcessRegressor(kernel=kernel1).fit(des1, vecs1)
kernel2 = ConstantKernel(1.65) * RBF(length_scale=7) + WhiteKernel() * ConstantKernel(0.6)
gpr3 = GaussianProcessRegressor(kernel=kernel2, random_state=0).fit(des2, vecs2)
print(gpr.score(des,vecs))
print(gpr2.score(des1,vecs1))
print(gpr3.score(des2,vecs2))
print(gpr.predict(des,return_std=True))
print(gpr2.predict(des1,return_std=True))
print(gpr3.predict(des2,return_std=True))
print(gpr.predict(a,return_std=True))
print(gpr2.predict(a,return_std=True))
print(gpr3.predict(a,return_std=True))
print(gpr.log_marginal_likelihood(gpr.kernel_.theta))
print(gpr2.log_marginal_likelihood(gpr.kernel_.theta))
print(gpr3.log_marginal_likelihood(gpr.kernel_.theta))

a=[1,2,3,4]
b=[0,2]
c=[a[i] for i in b]
print(c)