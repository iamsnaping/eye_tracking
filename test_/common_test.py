import copy

import cv2
import numpy as np
from eye_utils import data_util as du
import os
from pupil_detect import contour_detector as cd

root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.9\\left\\'
pic_path = os.path.join(root_path, '8.png')
img=cv2.imread(pic_path)


params = cd.PuRe_params()
params.threshold1=30
params.threshold2=40
detector=cd.PuRe(params=params)
sub=copy.deepcopy(params)
res=detector.detect(img=img)
print(res)
img=cd.draw_ellipse(img,res[0])
img=cd.draw_ellipse(img,[res[1]])
du.show_ph(img)

# center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
# axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
# drawed_img = cv2.ellipse(drawed_img, [center, axes, ellipse[2]], color=(0, 255, 0))
