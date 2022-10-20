import copy

import cv2
import numpy as np
import math
from eye_utils import data_util as du
import os
from pupil_detect import contour_detector as cd
from pupil_detect import contour_detector_single as cds
# du.get_data('C:\\Users\\snapping\\Desktop\\data\\2022.10.20\\')
# breakpoint()
root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.20\\origin\\'
pic_path = os.path.join(root_path, '387.png')
img=cv2.imread(pic_path)
t_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# p_img=cv2.threshold(t_img,50,255,cv2.THRESH_BINARY_INV)[1]
# g_img=cv2.threshold(t_img,120,255,cv2.THRESH_BINARY)[1]
# du.show_ph(p_img)
# du.show_ph(g_img)
# breakpoint()
gray_img=du.get_gray_pic(img)
gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 1)
pupil_img=cv2.threshold(gaussian_img,50,255,cv2.THRESH_BINARY_INV)[1]
# du.show_ph(pupil_img)
res1=cv2.connectedComponentsWithStatsWithAlgorithm(pupil_img,connectivity=8,ltype=cv2.CV_32S,ccltype=cv2.CCL_DEFAULT)
pupils=[]



for contours,centroid in zip(res1[2],res1[3]):
    if not 40<contours[2]<60:
        continue
    if not 40<contours[3]<60:
        continue
    min_s=contours[2]*contours[3]
    if contours[4]<min_s*0.6:
        continue
    # for i in range(contours[2]):
    #     for j in range(contours[3]):
    #         pupil_img[contours[1]+j][contours[0]+i]=125
    x=contours[0]+contours[2]/2
    y=contours[1]+contours[3]/2
    print(contours)
    dis=((x-centroid[0])**2)+((y-centroid[1])**2)
    pupils.append(centroid)
print(pupils)
img_1=img[int(pupils[0][1])-100:int(pupils[0][1])+100,int(pupils[0][0])-100:int(pupils[0][0])+100]
img_2=img[int(pupils[1][1])-100:int(pupils[1][1])+100,int(pupils[1][0])-100:int(pupils[1][0])+100]
img_1_origin=(int(pupils[0][0])-100,int(pupils[0][1])-100)
img_2_origin=(int(pupils[1][0])-100,int(pupils[1][1])-100)
# du.show_ph(img_1)
# du.show_ph(img_2)

# res2=cv2.connectedComponentsWithStatsWithAlgorithm(pupil_img,connectivity=8,ltype=cv2.CV_32S,ccltype=cv2.CCL_DEFAULT)
canny_img = cv2.Canny(gaussian_img, 30, 60)


params = cds.PuRe_params()
params.r_th=0.2
params.threshold1=30
params.threshold2=60
params.r_th=0.2
params.find_contour_param=cv2.CHAIN_APPROX_NONE
params.gd_max=20
params.glints_num=5
detector=cds.PuRe(params=params)
# sub=copy.deepcopy(params)
res=detector.detect(img=img_1)
img=cd.draw_ellipse(img,res[0],img_1_origin)
img=cd.draw_ellipse(img,[res[1]],img_1_origin)
res=detector.detect(img=img_2)
img=cd.draw_ellipse(img,res[0],img_2_origin)
img=cd.draw_ellipse(img,[res[1]],img_2_origin)
du.show_ph(img)

# center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
# axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
# drawed_img = cv2.ellipse(drawed_img, [center, axes, ellipse[2]], color=(0, 255, 0))
