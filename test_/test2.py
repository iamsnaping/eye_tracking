import copy
import functools

import cv2
import numpy as np
import math
from eye_utils import data_util as du
import os
from pupil_detect import contour_detector as cd
from pupil_detect import contour_detector_single as cds
from pupil_detect import  contour_detector_single_debug as cdsd

root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.21.2\\'

#
# du.get_data(root_path)
# breakpoint()

origin_path=os.path.join(root_path,'origin')
processed_path=os.path.join(root_path,'processed')
pass_path=os.path.join(root_path,'pass')
video_path=os.path.join(root_path,'video_path')
if not os.path.exists(video_path):
    os.mkdir(video_path)


params = cds.PuRe_params()
params.r_th=0.2
params.threshold1=30
params.threshold2=60
params.r_th=0.2
params.find_contour_param=cv2.CHAIN_APPROX_NONE
params.gd_max=20
params.gd_min=1
params.glints_num=5
params.pd_min=30
params.pd_max=60
params.p_binary_threshold=30
detector=cds.PuRe(params=params)

#debug

debug_detector=cdsd.PuRe(params)
img_path=os.path.join(origin_path,'228.png')
origin_img = cv2.imread(img_path)
gray_img=du.get_gray_pic(origin_img)
res=debug_detector.detect(gray_img)
breakpoint()

# sub=copy.deepcopy(params)

# if not os.path.exists(pass_path):
#     os.mkdir(pass_path)
# if not os.path.exists(processed_path):
#     os.mkdir(processed_path)
# img_list=os.listdir(origin_path)
# t=0
# for img_p in img_list:
#     print(img_p)
#     img_path=os.path.join(origin_path,img_p)
#     img_g = cv2.imread(img_path)
#     img_1=du.get_gray_pic(img_g)
#     res=detector.detect(img=img_1)
#     if isinstance(res,bool):
#         pass_img=os.path.join(video_path,img_p)
#         cv2.imwrite(pass_img,img_g)
#         t+=1
#         continue
#     # print(len(res[0]))
#     img_d=cds.draw_ellipse(img_g,res[0])
#     img_d=cds.draw_ellipse(img_d,res[1])
#     # du.show_ph(img_d)
#     d_path=os.path.join(video_path,img_p)
#     cv2.imwrite(d_path,img_d)
# print(f'total not pass {t}')


# center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
# axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
# drawed_img = cv2.ellipse(drawed_img, [center, axes, ellipse[2]], color=(0, 255, 0))
# def cmp1(x,y):
#     if len(x)<len(y):
#         return -1
#     elif len(x)>len(y):
#         return 1
#     return x<y
# def img2video():
#     fps = 30  # 视频每秒10帧
#     size = (1920, 1080)  # 需要转为视频的图片的尺寸
#     # 可以使用cv2.resize()进行修改
#
#     video = cv2.VideoWriter("C:\\Users\\snapping\\Desktop\\data\\2022.10.21.2\\video.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, size,True)
#     # 视频保存在当前目录下
#     filelist = os.listdir(video_path)
#     filelist.sort(key=functools.cmp_to_key(cmp1))
#
#     for item in filelist:
#         # print(item)
#         item = os.path.join(video_path,item)
#         img = cv2.imread(item)
#         # print(img.shape)
#         video.write(img)
#     video.release()
#
#
# img2video()



