import os
import test_main as tm

from base_estimation.plcr.plcr import *
from pupil_detect import contour_detector_single as cds
from pupil_detect import contour_detector_single_debug as cdsd
import cv2
from eye_utils import data_util as du


root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.22\\'
pic_path=os.path.join(root_path,'0.png')
img=cv2.imread(pic_path)

params = cds.PuRe_params()
params.r_th=0.2
params.threshold1=30
params.threshold2=60
params.r_th=0.2
params.find_contour_param=cv2.CHAIN_APPROX_NONE
params.gd_max=20
params.gd_min=1
params.glints_num=5
params.pd_min=40
params.pd_max=70
params.p_binary_threshold=30

# debug_detector=cdsd.PuRe(params)
# img_path=os.path.join(root_path,'0.png')
# origin_img = cv2.imread(img_path)
# gray_img=du.get_gray_pic(origin_img)
# res=debug_detector.detect(gray_img)
# breakpoint()

# detector=cds.PuRe(params=params)
# gray_img=du.get_gray_pic(img)
# res=detector.detect(img=gray_img)
# print(len(res[0]))
# print('glints')
# for contour in res[0]:
#     print(contour.ellipse)

# print('pupils')
# for contour in res[1]:
#     print(contour.ellipse)
eye_nums=[1577,691,1545,656,1604,654,1602,683,1545,656,1594,166]
es_module=plcr(1920,1080)
res=tm.get_estimate(eye_nums,es_module)
print(res)
'''
((1577, 691), (5.049665927886963, 5.9213433265686035), 269.983642578125)
((1550, 687), (5.4135003089904785, 6.358386516571045), 236.9146728515625)
((1602, 683), (4.84174919128418, 4.844640254974365), 135.0)
((1545, 656), (6.542692184448242, 7.049057960510254), 208.11669921875)
((1604, 654), (5.1743645668029785, 5.918323516845703), 13.521291732788086)


((1594, 656), (55.009178161621094, 60.308319091796875), 116.91022491455078)
'''

