import os
import numpy as np

from base_estimation.plcr import  plcr as bp
from pupil_detect import contour_detector_single as cds
import cv2
from pupil_detect import contour_detector_single_debug as cdsd
from eye_utils import data_util as du


root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.26\\wtc1\\cali'
root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.27\\wtc1\\glasses\\cali'
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\wq'
pic_path=os.path.join(root_path,'2.png')
img=cv2.imread(pic_path)

params = cds.PuRe_params()
params.threshold1=30
params.threshold2=60
params.r_th=0.3
params.find_contour_param=cv2.CHAIN_APPROX_NONE
params.gd_max=20
params.gd_min=1
params.glints_num=5
params.pd_min=30
params.pd_max=80
params.g_threshold=30
params.p_binary_threshold=50
params.g_binary_threshold=150


# nums=[19,2,21,23,28,29,3,30,5]
nums=[0,7]
skip_nums=[19]
#
debug_detector=cdsd.PuRe(params)
# debug_detector=cds.PuRe(params)
if __name__=='__main__':
    for i in nums:
        if i in skip_nums:
            continue
        img_path=os.path.join(root_path,str(i)+'.png')
        print(img_path)
        origin_img = cv2.imread(img_path)
        gray_img=du.get_gray_pic(origin_img)
        res=debug_detector.detect(gray_img)
        img=cdsd.draw_ellipse(origin_img,res[0][0])
        img=cdsd.draw_ellipse(img,res[1][0])
        print(len(res[0][0]))
        print(len(res[1][0]))
        du.show_ph(img,name=str(i))
        # breakpoint()
    breakpoint()
def get_glints_sort(glints):
    glints.sort(key=lambda x: x[1])
    sub_glints=glints[2:5]
    sub_glints.sort(key=lambda x:x[0])
    if glints[0][0]<glints[1][0]:
        glints[0],glints[1]=glints[1],glints[0]
    glints[2],glints[3],glints[4]=sub_glints[0],sub_glints[2],sub_glints[1]


class eye_tracker:
    def __init__(self):
        params = cds.PuRe_params()

        params.threshold1 = 30
        params.threshold2 = 60
        params.r_th = 0.3
        params.find_contour_param = cv2.CHAIN_APPROX_NONE
        params.gd_max = 20
        params.gd_min = 1
        params.glints_num = 5
        params.pd_min = 30
        params.pd_max = 80
        params.g_threshold = 30
        params.p_binary_threshold = 50
        params.g_binary_threshold=150

        self.pure=cds.PuRe(params)
        self.plcr=[bp.plcr(52.78,31.26),bp.plcr(52.78,31.26)]
        self.plcr[0]._rt = 220
        self.plcr[0]._radius = 0.78
        self.plcr[1]._rt = 220
        self.plcr[1]._radius = 0.78
    def set_params(self,params):
        self.pure.set_params(params)

    def set_calibration(self,vec):
        self.plcr[0].set_calibration(vec[0])
        self.plcr[1].set_calibration(vec[1])

    def get_estimation(self,num,eye_num):
        self.plcr[eye_num]._pupil_center = np.array([num[0][0], num[0][1], 0]).reshape((3, 1))
        self.plcr[eye_num]._param = np.array([0, 0, 0.62], dtype=np.float64).reshape((3, 1))
        self.plcr[eye_num].get_param()
        self.plcr[eye_num]._up = np.array([0, 1, 0], dtype=np.float64).reshape((3, 1))
        light = np.array(
            [num[1][0], num[1][1], 0, num[2][0], num[2][1], 0, num[3][0], num[3][1], 0, num[4][0], num[4][1], 0],
            dtype=np.float64).reshape((4, 3))
        light = light.T
        self.plcr[eye_num]._glints = self.plcr[eye_num]._pupil_center - light
        self.plcr[eye_num]._g0 = np.array([num[5][0], num[5][1], 0], dtype=np.float64).reshape((3, 1))
        self.plcr[eye_num]._g0 = self.plcr[eye_num]._pupil_center - self.plcr[eye_num]._g0
        self.plcr[eye_num].get_e_coordinate()
        self.plcr[eye_num].transform_e_to_i()
        self.plcr[eye_num].get_plane()
        self.plcr[eye_num].get_visual()
        self.plcr[eye_num].get_m_points()
        return self.plcr[eye_num].gaze_estimation()

    #binary img
    def detect(self,img,img_p=None):
        res = self.pure.detect(img=img)
        if isinstance(res,bool):
            return False
        glints_l = []
        glints_r=[]
        sub_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(res[0][0])<5 or len(res[1][0])<5:
            return False
        f=True
        if len(res[0][0])==5:
            for c in res[0][0]:
                glints_l.append(c.ellipse[0])
            f=False
            draw_img = cds.draw_ellipse(sub_img, res[0][0])
            draw_img = cds.draw_ellipse(draw_img, [res[0][1]])
            draw_img = cv2.circle(draw_img, (int(res[0][1].ellipse[0][0]), int(res[0][1].ellipse[0][1])), 3,(0, 255, 0), -1)
        if len(res[1][0])==5:
            for c in res[1][0]:
                glints_r.append(c.ellipse[0])
            if f:
                draw_img = cds.draw_ellipse(sub_img, res[0][0])
                draw_img = cds.draw_ellipse(draw_img, [res[0][1]])
                draw_img = cv2.circle(draw_img, (int(res[1][1].ellipse[0][0]), int(res[1][1].ellipse[0][1])), 3,(0, 255, 0), -1)
            else:
                draw_img = cds.draw_ellipse(draw_img, res[1][0])
                draw_img = cds.draw_ellipse(draw_img, [res[1][1]])
                draw_img = cv2.circle(draw_img, (int(res[1][1].ellipse[0][0]), int(res[1][1].ellipse[0][1])), 3,(0, 255, 0), -1)
        cv2.imwrite(img_p,draw_img)
        if len(glints_r)==0:
            get_glints_sort(glints_l)
            glints_r=glints_l.copy()
            left = [res[0][1].ellipse[0]]
            right=left.copy()
        elif len(glints_l)==0:
            get_glints_sort(glints_r)
            glints_l=glints_r.copy()
            right = [res[1][1].ellipse[0]]
            left=right.copy()
        else:
            get_glints_sort(glints_l)
            get_glints_sort(glints_r)
            left = [res[0][1].ellipse[0]]
            right=[res[1][1].ellipse[0]]
        left.extend(glints_l)
        right.extend(glints_r)
        l=self.get_estimation(left,0)
        self.plcr[0].refresh()
        r=self.get_estimation(right,1)
        self.plcr[1].refresh()
        return l,r
