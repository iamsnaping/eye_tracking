import math
import os
import test_main as tm
import cv2
import numpy as np

from eye_tracking import eye_tracking as et
from eye_utils import data_util as du

# root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.22\\'
# root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.24\\origin'
# root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.24\\up'
# root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\origin'
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\origin_2'
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\test'
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\wq'
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\test2'
# root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\test3'
root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.25\\test4'
root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.26\\wtc1'
root_path='C:\\Users\\snapping\\Desktop\\data\\2022.10.27\\wtc1\\glasses'
pic_path=os.path.join(root_path,'drawed')
if not os.path.exists(pic_path):
    os.makedirs(pic_path)
bias_num=5
'''
数据1:
        最小偏移             平均偏移            最大偏移
右眼偏移:0.1906901967184051 3.775177889852158 7.299037145484567
左眼偏移:0.6071588985018462 2.9347389905244876 5.897104509395121
两眼偏移:0.7093625477544381 3.137615118750815 6.090978899482407
修正向量:[-0.62839237 -3.0056655 ] [ 0.42103756 -2.77069682]
数据2:
        最小偏移             平均偏移            最大偏移
右眼偏移:1.6727797827271393 3.5960741499344993 5.965603592726228
左眼偏移:2.3021653746649045 3.841392391184704 5.965755237239784
两眼偏移:1.9385601987976164 3.6232042132107756 5.951912900484634
修正向量:[ 0.97432124 -4.95442728] [ 2.85953707 -4.80431618]
数据3:
        最小偏移             平均偏移            最大偏移
右眼偏移:0.8206306748541384 2.7199453856354654 4.8275975792198045
左眼偏移:0.785950505838765 3.309310352676106 4.415107819052943
两眼偏移:0.7640980130222771 2.9274544451849076 4.380313371082332
[-0.48848684 -7.73302188] [ 1.92241002 -5.02935555]
数据4:
        最小偏移             平均偏移            最大偏移
右眼偏移:0.3515615403607752 2.394904424158114 5.842185579697135
左眼偏移:0.5201670089298216 2.886736038293786 8.413450940117029
两眼偏移:0.13871791551426982 2.4717762052499954 5.5144497164610256
修正向量:[-0.22814999 -6.83937852] [ 2.84926039 -3.3567826 ]
数据5:
        最小偏移             平均偏移            最大偏移
右眼偏移:0.6039262484559265 2.68645816649484 4.976219226802376
左眼偏移:1.1740765462582143 2.9467538965452857 5.444247546512259
两眼偏移:0.3170329944156145 2.58177645934719 4.4193555198337355
修正向量:[ 0.17731257 -6.13094211] [ 1.82032898 -4.72469139]
'''
def get_path(dir):
    global root_path
    r_path=os.path.join(root_path,dir)
    p_path=os.path.join(pic_path,dir)
    if not os.path.exists(r_path):
        os.makedirs(r_path)
    if not os.path.exists(p_path):
        os.makedirs(p_path)
    d_list=os.listdir((r_path))
    t_file=[]
    i_file=[]
    images=[]
    for d in d_list:
        if os.path.splitext(d)[1]=='.txt':
            t_file.append(os.path.join(r_path,d))
        else:
            images.append(os.path.join(p_path,d))
            i_file.append(os.path.join(r_path,d))
    return t_file,i_file,images

def main():
    vec_left,vec_right,vec_ave=cali()
    a=[[0,0] for i in range(9)]
    a=np.array(a,dtype=np.float64)
    b=a.copy()
    tracker_1=et.eye_tracker()
    # tracker_1.set_calibration([vec_left,vec_right])
    tracker_2=et.eye_tracker()
    tracker_2.set_calibration([a,b])
    vec_left=np.zeros(2,dtype=np.float64)
    vec_right=np.zeros(2,dtype=np.float64)
    left=[]
    right=[]
    ave=[]
    vec_ave=np.zeros(2,dtype=np.float64)
    tt=0
    img_ps=[]
    txts,imgs,image_ps=get_path('test')

    for txt_p,img_p,image_p in zip(txts,imgs,image_ps):
        # print(img_p)
        img=cv2.imread(img_p)
        gray_img=du.get_gray_pic(img)
        res=tracker_1.detect(gray_img,image_p)
        res2=tracker_2.detect(gray_img,image_p)
        if isinstance(res,bool):
            print(img_p,end=' ')
            print('pass')
            continue
        with open(txt_p) as t:
            txt=t.read()
        txt=txt.split(' ')
        img_ps.append(img_p)
        txt=np.array(txt,dtype=np.float64)
        print('this is compare',res,txt*52.78/1920,image_p)
        ave.append((res[1]+res[0])/2-txt*52.78/1920)
        left.append(res[0]-txt*52.78/1920.0)
        right.append(res[1]-txt*52.78/1920.0)
    # vec_left/=tt
    # vec_right/=tt
    # vec_ave/=tt
    for a,b,c in zip(left,right,ave):
        print(a,b,c)
    # ave_left,ave_right,max_left,min_left,max_right,min_left,total=0,0,0,0,0,0,0
    # max_left,max_right=-1,-1
    # min_left,min_right=10,10
    # ave_a,max_a,min_a=0,-1,10
    # ave_cc,max_cc,min_cc=0,-1,10
    # for l,r,aver,img_p in zip(left,right,ave,img_ps):
    #     a=l-vec_left
    #     b=r-vec_right
    #     print(a,b,end=' ')
    #     c=(a+b)/2
    #     d=aver-vec_ave
    #     a=math.sqrt((a**2).sum())
    #     b=math.sqrt((b**2).sum())
    #     c=math.sqrt((c**2).sum())
    #     d=math.sqrt((d**2).sum())
    #     ll=math.asin(a/80)*180/math.pi
    #     rr=math.asin(b/80)*180/math.pi
    #     cc=math.asin(c/80)*180/math.pi
    #     aa=math.asin(d/80)*180/math.pi
    #     ave_left+=ll
    #     ave_right+=rr
    #     ave_cc+=cc
    #     ave_a+=aa
    #     max_a=max(max_a,aa)
    #     min_a=min(min_a,aa)
    #     max_cc=max(cc,max_cc)
    #     min_cc=min(cc,min_cc)
    #     max_left=max(ll,max_left)
    #     min_left=min(ll,min_left)
    #     max_right=max(rr,max_right)
    #     min_right=min(rr,min_right)
    #     total+=1
    #     print(ll,end=' ')
    #     print(rr,end=' ')
    #     print(img_p)
    # print('        最小误差                 平均误差                 最大误差')
    # print(f'左眼误差：{min_left,ave_left/total,max_left}')
    # print(f'有眼无差：{min_right, ave_right / total, max_right}')
    # print(f'两眼均值误差{min_cc,ave_cc/total,max_cc}')
    # print(min_a,ave_a/total,max_a)

def cali():
    tracker=et.eye_tracker()
    vecs_left=[]
    vecs_right=[]
    vecs_ave=[]
    txts,imgs,image_ps=get_path('cali')

    for txt_p,img_p,image_p in zip(txts,imgs,image_ps):
        img=cv2.imread(img_p)
        gray_img=du.get_gray_pic(img)
        # tracker.set_calibration([np.array([[0,0] for i in range(9)],dtype=np.float64),np.array([[0,0] for i in range(9)],dtype=np.float64)])
        res=tracker.detect(gray_img,image_p)
        # print(img_p,res)
        if isinstance(res,bool):
            print('cali',img_p)
            continue
        with open(txt_p) as t:
            txt=t.read()
        txt=txt.split(' ')
        txt=np.array(txt,dtype=np.float64)
        txt=txt*51.78/1920
        vecs_left.append(res[0]-txt)
        vecs_right.append(res[1]-txt)
        vecs_ave.append((res[0]+res[1])/2-txt)
    print('vecs begin')
    for a,b,c in zip(vecs_left,vecs_right,vecs_ave):
        print(a,b,c)
    print('vecs end')
    print('cali end')
    return np.array(vecs_left,dtype=np.float64),np.array(vecs_right,dtype=np.float64),np.array(vecs_ave,dtype=np.float64)




if __name__=='__main__':
    # cali()
    main()

