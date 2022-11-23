
import os
import threading

import numpy as np
import pynput.keyboard

from base_estimation.plcr import  plcr as bp
from pupil_detect import contour_detector_single as cds
import cv2
from pupil_detect import contour_detector_single_debug as cdsd
from eye_utils import data_util as du
import timeit
import pyautogui
import multiprocessing
import queue
import sys
from pynput import keyboard


STOP_FLAG=False

EXIT_FLAG=False


def on_press(key):
    if isinstance(key,pynput.keyboard.KeyCode):
        if key.char=='q':
            global EXIT_FLAG
            EXIT_FLAG=True
            return False

q=queue.Queue(30)

root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.26\\wtc1\\cali'
root_path='C:\\Users\\snapping\\Desktop\\data\\2022.11.8\\wtc\\cali\\0'
from data_geter.data_geter import *
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\wq'
pic_path=os.path.join(root_path,'0.png')
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
params.p_binary_threshold=45
params.g_binary_threshold=145

time_recoder=np.array([0.,0.])
times_recorder=np.array([0.,0.])

def get_pics(cap):
    while True:
        ref,frame=cap.read()
        if STOP_FLAG:
            break
        if not q.full():
            q.put(frame)


def clear_q(t):
    global STOP_FLAG
    STOP_FLAG=True
    t.join()
    print(q.qsize())
    while not q.empty():
        q.get()


def record_time(type):
    def decorator(func):
        def inner():
            global time_recoder
            start=timeit.default_timer()
            func()
            end=timeit.default_timer()
            time_recoder[type]+=end-start
            return func
        return inner
    return decorator


# nums=[19,2,21,23,28,29,3,30,5]
nums=[0,3,4]
skip_nums=[19]
#
root_path='C:\\Users\\snapping\\Desktop'
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
        img=cdsd.draw_ellipse(img,[res[0][1]])
        img=cdsd.draw_ellipse(img,[res[1][1]])
        for i in res[0][0]:
            print(i.ellipse,end=' ')
        print('')
        for i in res[1][0]:
            print(i.ellipse,end=' ')
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
        params.r_th = 0.5
        params.find_contour_param = cv2.CHAIN_APPROX_NONE
        params.gd_max = 20
        params.gd_min = 2
        params.glints_num = 5
        params.pd_min = 30
        params.pd_max = 60
        params.g_threshold = 30
        params.p_binary_threshold = 45
        params.g_binary_threshold = 140
        self.is_calibration=True
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
        params.p_binary_threshold = 45
        params.g_binary_threshold = 145
        self.average_mode=False

        self.pure=cds.PuRe(params)
        self.plcr=[bp.plcr(52.78,31.26),bp.plcr(52.78,31.26)]
        self.plcr[0]._rt = 220
        self.plcr[0]._radius = 0.78
        self.plcr[1]._rt = 220
        self.plcr[1]._radius = 0.78
    def set_params(self,params):
        self.pure.set_params(params)

    def set_calibration(self,vec,des):
        if len(vec)==2:
            self.plcr[0].set_calibration(vec[0],des[0])
            self.plcr[1].set_calibration(vec[1],des[1])
        else:
            self.plcr[0]._is_calibration=False
            self.plcr[1]._is_calibration = False
            self.average_mode=True
            self.plcr[0].set_calibration(vec[0], des[0])
            self.plcr[1].set_calibration(vec[0], des[0])
            self.vecs=vec[0]
            self.des=des[0]
    def get_estimation(self,num,eye_num):
        self.plcr[eye_num]._pupil_center = np.array([num[0][0], num[0][1], 0]).reshape((3, 1))
        self.plcr[eye_num]._param = np.array([0, 0, 0.62], dtype=np.float32).reshape((3, 1))
        self.plcr[eye_num].get_param()
        self.plcr[eye_num]._up = np.array([0, 1, 0], dtype=np.float32).reshape((3, 1))
        light = np.array(
            [num[1][0], num[1][1], 0, num[2][0], num[2][1], 0, num[3][0], num[3][1], 0, num[4][0], num[4][1], 0],
            dtype=np.float32).reshape((4, 3))
        light = light.T
        self.plcr[eye_num]._glints = self.plcr[eye_num]._pupil_center - light
        self.plcr[eye_num]._g0 = np.array([num[5][0], num[5][1], 0], dtype=np.float32).reshape((3, 1))
        self.plcr[eye_num]._g0 = self.plcr[eye_num]._pupil_center - self.plcr[eye_num]._g0
        self.plcr[eye_num].get_e_coordinate()
        self.plcr[eye_num].transform_e_to_i()
        self.plcr[eye_num].get_plane()
        self.plcr[eye_num].get_visual()
        self.plcr[eye_num].get_m_points()
        return self.plcr[eye_num].gaze_estimation()
    def calibration_mode(self,mode):
        self.is_calibration=mode
        self.plcr[0]._is_calibration=mode
        self.plcr[1]._is_calibration=mode
    #binary img
    def detect(self,img,img_p=None):
        global time_recoder,times_recorder
        time1=timeit.default_timer()
        res = self.pure.detect(img=img)
        time2=timeit.default_timer()
        times_recorder[0]+=1.
        time_recoder[0]+=time2-time1
        if isinstance(res,bool):
            return 'can not detect glints or pupils'
        glints_l = []
        glints_r=[]
        sub_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if len(res[0][0])<5 or len(res[1][0])<5:
            return 'glints is not enough'
        f=True
        if len(res[0][0])==5:
            for c in res[0][0]:
                glints_l.append(c.ellipse[0])

            f=False
            '''
            draw_img = cds.draw_ellipse(sub_img, res[0][0])
            draw_img = cds.draw_ellipse(draw_img, [res[0][1]])
            draw_img = cv2.circle(draw_img, (int(res[0][1].ellipse[0][0]), int(res[0][1].ellipse[0][1])), 3,(0, 255, 0), -1)
            '''
        if len(res[1][0])==5:
            for c in res[1][0]:
                glints_r.append(c.ellipse[0])
        '''
            if f:
                draw_img = cds.draw_ellipse(sub_img, res[0][0])
                draw_img = cds.draw_ellipse(draw_img, [res[0][1]])
                draw_img = cv2.circle(draw_img, (int(res[1][1].ellipse[0][0]), int(res[1][1].ellipse[0][1])), 3,(0, 255, 0), -1)
            else:
                draw_img = cds.draw_ellipse(draw_img, res[1][0])
                draw_img = cds.draw_ellipse(draw_img, [res[1][1]])
                draw_img = cv2.circle(draw_img, (int(res[1][1].ellipse[0][0]), int(res[1][1].ellipse[0][1])), 3,(0, 255, 0), -1)
        cv2.imwrite(img_p,draw_img)
        '''

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
        '''
        file_path=os.path.dirname(img_p)+'\\'+os.path.basename(img_p).split('.')[0]+'.txt'
        with open(file_path,'w') as t:
            for ce in glints_l:
                t.write(str(ce)+'\n')
            t.write(str(res[0][1].ellipse[0])+'\n\n')
            for ce in glints_r:
                t.write(str(ce)+'\n')
            t.write(str(res[1][1].ellipse[0]))
        '''
        left.extend(glints_l)
        right.extend(glints_r)
        compute_vec = np.zeros(2, dtype=np.float32)
        dis_vec = np.array([0., 0.], dtype=np.float32)
        l=self.get_estimation(left,0)
        self.plcr[0].refresh()
        r=self.get_estimation(right,1)
        self.plcr[1].refresh()
        g_len=(l[0]+r[0])/2
        centers = np.array([52.78 / 2, 31.26 / 2], dtype=np.float32)
        # print(f'glen {g_len}')
        gaze_estimation=l*(52.78-g_len)/52.78+r*g_len/52.78
        #vec direction->down
        # up down left right
        time3=timeit.default_timer()
        if self.average_mode:
            if self.is_calibration:
                return gaze_estimation
            centers=np.array([self.des[0][0],self.des[0][1]],dtype=np.float32)
            # centers = np.array([52.78 / 2, 31.26 / 2], dtype=np.float32)
            s_para = 52.78 / 1920
            compute_vec = np.zeros((2), dtype=np.float32)
            # y 480,960 x 660 1320
            # up
            # vec1 = (self.vecs[0] - self.vecs[2]) / (480 * s_para)
            vec1=(self.vecs[0]-self.vecs[2])
            # down
            # vec2 = (self.vecs[5] - self.vecs[0]) / (480 * s_para)
            vec2=(self.vecs[5] - self.vecs[0])
            # up
            # vec3 = (self.vecs[4] - self.vecs[1]) / (960 * s_para)
            # vec4 = (self.vecs[6] - self.vecs[3]) / (960 * s_para)
            vec3=(self.vecs[4] - self.vecs[1])
            vec4 = (self.vecs[6] - self.vecs[3])
            centers_ratio=norm(self.des[0]-self.des[2])/(norm(self.des[5]-self.des[0])+norm(self.des[0]-self.des[2]))
            if centers[0]>gaze_estimation[0]:
                left_vec=self.des[1]*(1-centers_ratio)+self.des[4]*centers_ratio
                ratio=norm(gaze_estimation-self.des[1])/(norm(self.des[2]-gaze_estimation)+norm(gaze_estimation-self.des[1]))
                mid1=ratio*self.des[2]+(1-ratio)*self.des[1]
                ratio=norm(gaze_estimation-left_vec)/(norm(gaze_estimation-left_vec)+norm(gaze_estimation-centers))
                mid2=ratio*self.des[0]+(1-ratio)*left_vec
                ratio=norm(gaze_estimation-self.des[4])/(norm(gaze_estimation-self.des[4])+norm(gaze_estimation-self.des[5]))
                mid3=ratio*self.des[5]+(1-ratio)*self.des[4]
                scal_1=norm(gaze_estimation-mid1)/norm(mid3-mid1)
            else:
                right_vec=self.des[3]*(1-centers_ratio) + self.des[6] * centers_ratio
                ratio=norm(gaze_estimation - self.des[2])/(norm(self.des[2] - gaze_estimation) + norm(gaze_estimation - self.des[3]))
                mid1 = ratio * self.des[3]+(1-ratio)*self.des[2]
                ratio=norm(gaze_estimation - centers) /(norm(gaze_estimation - right_vec) + norm(gaze_estimation - centers))
                mid2 = ratio*right_vec+(1-ratio)*centers
                ratio=norm(gaze_estimation - self.des[5]) / (norm(gaze_estimation - self.des[5]) + norm(gaze_estimation - self.des[6]))
                mid3 = ratio* self.des[6]+(1-ratio)*self.des[5]
                scal_1=np.linalg.norm(gaze_estimation-mid1)/np.linalg.norm(mid3-mid1)
            if gaze_estimation[1] < mid2[1]:
                scal_2=np.linalg.norm(gaze_estimation-mid1)/np.linalg.norm(mid2-mid1)
                vec5 = scal_2* vec1 + self.vecs[2]
                vec5_des=self.des[2]+(self.des[0]-self.des[2])*scal_2
            else:
                scal_2=np.linalg.norm(gaze_estimation-mid2)/np.linalg.norm(mid3-mid2)
                vec5 = scal_2* vec2 + self.vecs[0]
                vec5_des=self.des[0]+(self.des[5]-self.des[0])*scal_2
            if gaze_estimation[0] < centers[0]:
                vec6 = vec3 * scal_1 + self.vecs[1]
                vec6_des=(self.des[4]-self.des[1])*scal_1
                # compute_vec = (vec5 - vec6) / (660 * s_para) * gaze_estimation[0] + vec6
                compute_vec = (vec5 - vec6) / np.linalg.norm(vec6_des[0]-vec5_des[0]) * gaze_estimation[0] + vec6
            else:
                vec6 = vec4 * scal_1 + self.vecs[3]
                vec6_des=(self.des[6]-self.des[3])*scal_1
                # compute_vec = (vec6 - vec5) / (660 * s_para) * (gaze_estimation[0] - centers[0]) + vec5
                compute_vec = (vec6 - vec5) / np.linalg.norm(vec6_des[0]-vec5_des[0]) * (gaze_estimation[0] - centers[0]) + vec5
            # print(f'sca {scal_1, scal_2,mid1,mid2,mid3,gaze_estimation}')
            time3 = timeit.default_timer()
            time_recoder[1]+=time3-time2
            times_recorder[1]+=1.
            return gaze_estimation-compute_vec
        times_recorder[1]+=1
        time_recoder[1]+=time3-time2
        return l,r

    def calibration(self,cap,screen,points,cali_nums=20,store=False,root_path=None):
        self.calibration_mode(True)
        points_num=1
        vecs_left = []
        vecs_right = []
        vecs_ave = []
        des_left, des_right, des_ave = [], [], []
        background_color = (255, 255, 255)
        RED = (0, 0, 255)
        BLUE = (255, 0, 0)
        GREEN = (0, 255, 0)
        r_cali_path=os.path.join(root_path,'cali')
        r_draw_path=os.path.join(root_path,'draw')
        points=np.array(points,dtype=np.float32)
        for i in range(10):
            ref,frame=cap.read()
        for point in points:
            nums=0
            ress = []
            cali_path=os.path.join(r_cali_path,str(points_num))
            draw_path=os.path.join(r_draw_path,str(points_num))
            if not os.path.exists(cali_path):
                os.makedirs(cali_path)
            if not os.path.exists(draw_path):
                os.makedirs(draw_path)
            if store:
                points_path=os.path.join(root_path,str(points_num))
                if not os.path.exists(points_path):
                    os.makedirs(points_path)
                points_num+=1
            screen.fill(background_color)
            pygame.draw.circle(screen, RED, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)
            while True:
                pygame.draw.circle(screen, BLUE, (point[0], point[1]), 10)
                pygame.display.update()
                if nums>=cali_nums:
                    break
                ref, frame = cap.read()
                gray_img = du.get_gray_pic(frame)
                d_path=os.path.join(draw_path,str(nums)+'.png')
                res = self.detect(gray_img,img_p=d_path)
                if isinstance(res,str):
                    continue
                if store:
                    pass
                    pic_pa=os.path.join(points_path,str(nums)+'.png')
                    cv2.imwrite(pic_pa,frame)
                    file_path=os.path.join(points_path,str(nums)+'.txt')
                    content=str(point[0])+' '+str(point[1])
                    with open(file_path, 'w') as fd:
                        fd.write(str(content))
                        fd.close()
                ress.append(res)
                nums+=1
                print(nums)
            pygame.draw.circle(screen, GREEN, (point[0], point[1]), 10)
            pygame.display.update()
            time.sleep(1.0)

            res=[]
            average = np.mean(ress, axis=0)
            if len(ress) == 0:
                continue
            for r1 in ress:
                if not np.linalg.norm(r1 - average) > 10:
                    res.append(r1)
            if len(res) == 0:
                continue
            res = np.mean(res, axis=0)
            g_len = np.fabs((res[0][0] + res[1][0]) / 2)
            vecs_right.append((res[1] - point * 52.78 / 1920))
            vecs_left.append((res[0] - point * 52.78 / 1920))
            vecs_ave.append((res[0] * (52.78 - g_len) / 52.78) + (res[1] * (g_len) / 52.78) - point * 52.78 / 1920)
            des_left.append(res[0])
            des_right.append(res[1])
            des_ave.append((res[0] * (52.78 - g_len) / 52.78) + (res[1] * (g_len) / 52.78))
        pygame.quit()
        return np.array(vecs_left, dtype=np.float32), \
               np.array(vecs_right, dtype=np.float32), \
               np.array(vecs_ave, dtype=np.float32), \
               np.array(des_left, dtype=np.float32), np.array(des_right, dtype=np.float32), np.array(des_ave,
                                                                                                     dtype=np.float32)


    def tracking(self,cap,vec_left, vec_right,vec_ave,des_left, des_right,des_ave,store=False,root_path=None):
        pyautogui.FAILSAFE=False

        def run_key():
            with keyboard.Listener(on_press=on_press) as lsn:
                lsn.join()
        t=threading.Thread(target=run_key)
        t.start()
        if not (root_path is None):
            if os.path.exists(root_path):
                os.makedirs(root_path)
        self.set_calibration([vec_ave], [des_ave])
        # tracker_2.set_calibration([np.array([[0.,0.] for i in range(7)])],[np.array([[0.,0.] for i in range(7)])])
        self.calibration_mode(False)
        start_t=timeit.default_timer()
        point=np.zeros(2,dtype=np.float32)
        times=0.
        valid=0.
        data_t=0.
        pro_t=0.
        # thread_p=threading.Thread(target=get_pics,args=(cap,))
        # thread_p.setDaemon(True)
        # thread_p.start()
        global EXIT_FLAG
        while True:
            if EXIT_FLAG==True:
                break
            # print('a')
            if q.empty():
                t1=timeit.default_timer()
                ref, frame = cap.read()
                # print(frame.shape)
                # frame=q.get()
                # print(frame)
                t2=timeit.default_timer()
                gray_img=du.get_gray_pic(frame)
                res = self.detect(gray_img)
                t3 = timeit.default_timer()
                pro_t += t3 - t2
                data_t += t2 - t1
                times+=1
                if not isinstance(res, str):
                    valid+=1
                    u_point=res/52.78*1920
                    if np.linalg.norm(u_point-point)>30:
                        point=u_point
                print(res)
            pyautogui.moveTo(point[0],point[1])

            if valid==50:
                end_t=timeit.default_timer()
                total_t=end_t-start_t
                print(total_t)
                print(valid)
                print(times)
                print(pro_t,pro_t/times,pro_t/valid)
                print(data_t,data_t/times,data_t/valid)
                print(pro_t/total_t)
                print(data_t/total_t)
                # clear_q(thread_p)
                # print(q.qsize())
                # break





def norm(a):
    return np.linalg.norm(a)

def get_time():
    a=cds.get_time()
    return a,time_recoder,times_recorder,time_recoder/times_recorder,time_recoder/time_recoder.sum()