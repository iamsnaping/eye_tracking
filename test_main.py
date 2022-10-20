import math
import xml.sax.handler
import math
import xml.sax.handler

from base_estimation.plcr.plcr import plcr
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# 342.520325 256.411835
# 320.601532 221.878540
# 364.757599 220.257584
# 364.527313 251.839600
# 321.264709 253.619522
# 362.916962 220.244675 64.084343 74.064552 136.148544

# 272.941071 292.299927
# 250.682770 259.503143
# 294.485168 257.725494
# 293.811340 288.928192
# 251.993057 289.338593
# 255.961700 287.527557 74.085304 77.724098 110.296799
head=np.array([17.5,13.5,57.5])
head_bias=[[0,0,0],[0,0,12.5],[0,0,25],
           [-12.5,0,0],[-12.5,0,12.5],[-12.5,0,25],
           [-25,0,0],[-25,0,12.5],[-25,0,25]]
index=0



#             #################                 #################                 #################                 #################                 #################
#             #################                 #################                 #################                 #################                 #################
#             ##             ##                 ##             ##                 ##                                       ###                        ###
#             ##             ##                 ##             ##                 ##                                       ###                        ###
#             ##             ##                 ##             ##                 ##                                       ###                        ###
#             ##             ##                 ##             ##                 ##                                       ###                        ###
#             ##             ##                 ##             ##                 ##                                       ###                        ###
#             #################                 #################                 #################                        ###                        ###
#             #################                 #################                 #################                        ###                        ###
#             ##             ##                 ##             ##                                ##                        ###                        ###
#             ##             ##                 ##             ##                                ##                        ###                        ###
#             ##             ##                 ##             ##                                ##                        ###                        ###
#             ##             ##                 ##             ##                                ##                        ###                        ###
#             ##             ##                 ##             ##                                ##                        ###                        ###
#             #################                 ##             ##                 #################                 #################                 #################
#             #################                 ##             ##                 #################                 #################                 #################
#              @copyright
#              2022-2122
#                                                                                 *****************
#                                                                                 ** wtc warning **
#                                                                                 *****************
#                                                             Chinese law provides severe civil and criminal penalties for
#                                                             the unauthorized reproduction, distribution, or exhibition of
#                                                             copyrighted motion pictures(Title 22,People' s republic of China
#                                                             code, Section 369 and 443).The TC Bureau of investigation
#                                                             investigates all allegations of criminal copyright infringement
#                                                              (Title 22,Peole' s Republic of China Code,section 4396)
#
#
def test_all_one_side(path='D:\\download\\new_data_framework\\s1\\L\\', t=None):
    global index
    goal=get_goal()
    for i in range(9):
        r_path=os.path.join(path,str(i))
        index=i
        estimation=get_ave(r_path=r_path,plcr_=t)
        show_data(estimation,goal,i)


def test_all(path_l='D:\\download\\new_data_framework\\s1\\L\\',path_r='D:\\download\\new_data_framework\\s1\\R\\',t_l=None,t_r=None):
    goal=get_goal()
    for i in range(9):
        l_path=os.path.join(path_l,str(i))
        r_path=os.path.join(path_r,str(i))
        r_e=get_ave(r_path=r_path,plcr_=t_r)
        l_e=get_ave(r_path=l_path,plcr_=t_l)
        estimation=[]
        for l_points,r_points in zip(l_e,r_e):
            mid=[]
            for l,r in zip(r_points,r_points):
                mid.append((l+r)/2.0)
            estimation.append(mid)
        show_data(estimation,goal,i)


def get_ave(r_path=None,plcr_=None):
    list_dir=os.listdir(r_path)
    estimation=[]
    # print(list_dir)
    for dir in list_dir:
        dir_f=os.path.join(r_path,dir)
        l_d=os.listdir(dir_f)
        mid = []
        for f_d in l_d:
            d=os.path.join(dir_f,f_d)
            with open(d) as f:
                t=f.read()
                t=t.replace('\n',' ')
                t=t.split(' ')
                num=get_num(t)
                mid.append(get_estimate(num, es_module=plcr_))
            f.close()
        estimation.append(mid)
    return estimation


def show_data(estimation,goal,index):
    # ave=analyze_es(estimation)
    # get_mean_x_y(ave)
    mid = compute_bias(estimation, goal)
    bias, ave_vec = mid[0], mid[1]
    # for i in range(7):
    #     for j in range(7):
    #         print(bias[i+j*7],end=' ')
    #     print('')
    result = compute_bias_angle(estimation, goal)
    maxi = result[0]
    ave = result[1]
    mini = result[2]
    # print('max_error\tmean_error\tmin_error',end='\t')
    # print('')
    # for i in range(6,-1,-1):
    #     for j in range(7):
    #         print(f'{maxi[i*7+j]}\t{ave[i*7+j]}\t{mini[i*7+j]}\t')
    print(f'index: {index}', end='  ')
    print(f'{sum(maxi) / len(maxi)} {sum(ave) / len(ave)} {sum(mini) / len(mini)} {ave_vec[0][0]} {ave_vec[1][0]}')
    draw(estimation,goal)

def draw(estimation,ave):
    x=[]
    y=[]
    col=[]
    s=[]
    k=1
    for points in estimation:
        for point in points:
            x.append(point[0][0])
            y.append(point[1][0])
            col.append(k)
            s.append(2)
        k+=1
    for point in ave:
        x.append(point[0][0])
        y.append(point[1][0])
        col.append(k)
        k+=1
        s.append(5)
    # plt.plot(x,y,linewidth=2)
    plt.scatter(x,y,c=col)
    plt.show()

def analyze_es(estimation):
    ave_points=[]
    for points in estimation:
        mid_ave = np.zeros((2, 1), np.float64).reshape((2,1))
        t=0
        for point in points:
            # print(mid_ave)
            # print(point)
            # print(mid_ave+point)
            mid_ave=mid_ave+point
            t+=1
        mid_ave/=t
        ave_points.append(mid_ave)
    return ave_points

def get_mean_x_y(ave):
    x_mean_t=[]
    y_mean_t=[]
    for i in range(7):
        y_mean = 0.0
        x_mean = 0.0
        for j in range(1,7):
            x_mean+=(ave[i*7+j][0][0]-ave[i*7+j-1][0][0])
        x_mean/=6.0
        x_mean_t.append(x_mean)
        for j in range(0,6):
            y_mean+=(ave[i+(j+1)*7][1][0]-ave[i+j*7][1][0])
        y_mean/=6.0
        y_mean_t.append(y_mean)
    return x_mean_t,y_mean_t


def get_estimate(num, es_module=None):
    es_module._pupil_center = np.array([num[10], num[11], 0]).reshape((3, 1))
    es_module._param = np.array([0, 0, 0.42], dtype=np.float64).reshape((3, 1))
    es_module.get_param()
    es_module._up = np.array([0, 1, 0], dtype=np.float64).reshape((3, 1))
    light = np.array(
        [num[4], num[5], 0, num[2], num[3], 0, num[8], num[9], 0, num[6], num[7], 0],
        dtype=np.float64).reshape((4, 3))
    light = light.T
    es_module._glints = es_module._pupil_center - light
    es_module._g0 = np.array([num[0], num[1], 0], dtype=np.float64).reshape((3, 1))
    es_module._g0 = es_module._pupil_center - es_module._g0
    es_module.get_e_coordinate()
    es_module.transform_e_to_i()
    es_module.get_plane()
    es_module.get_visual()
    es_module.get_m_points()
    return es_module.gaze_estimation()

def get_num(s):
    num=[]
    for _ in s:
        num.append(trans(_))
    return num

def trans(s):
    m=1.0
    k=0.1
    t=0.0
    f=True
    for i in s:
        if i=='.':
            f=False
            m=0.1
        else:
            if f==True:
                t=t*m+int(i)
                if m==1:
                    m=10.0
            else:
                t=t+int(i)*m
                m *= k
    return t



def get_goal():
    x=5.0
    y=4.0
    x_margin=2.0
    y_margin=1.5
    points=[]
    for i in range(7):
        y_co = i * y + y_margin
        for j in range(7):
            x_co = x * j + x_margin
            point=np.array([x_co,y_co],np.float64).reshape((2,1))
            points.append(point)
    return points

def compute_bias(estimation,goal):
    ave_bias=[]
    ave_vec=np.array([0,0],np.float64).reshape((2,1))
    for i in range(49):
        bias=0.0
        for point in estimation[i]:
            bias=bias+np.linalg.norm(point-goal[i])
            ave_vec+=(point-goal[i])
        bias/=len(estimation[i])
        ave_bias.append(bias)
    ave_vec/=49.0*20.0
    return ave_bias,ave_vec

def compute_bias_angle(estimation,goal):
    global head
    global head_bias
    global index
    head=head+head_bias[index]
    ave_bias=[]
    max_bias=[]
    min_bias=[]
    vec_t=np.zeros(shape=(3),dtype=np.float64)
    vec_g=np.zeros(shape=(3),dtype=np.float64)
    for i in range(49):
        mini=100.0
        maxi=-1.0
        ave=0.0
        for point in estimation[i]:
            vec_g[0],vec_g[1]=point[0][0],point[1][0]
            vec_t[0],vec_t[1]=goal[i][0][0],goal[i][1][0]
            vec_g-=head
            vec_t-=head
            vec_g/=np.linalg.norm(vec_g)
            vec_t/=np.linalg.norm(vec_t)
            alpha=math.acos(np.dot(vec_g,vec_t))*180.0/math.pi
            mini=min(alpha,mini)
            maxi=max(alpha,maxi)
            ave+=alpha
            vec_g[2]=0.0
            vec_t[2]=0.0
        ave/=len(estimation[i])
        min_bias.append(mini)
        max_bias.append(maxi)
        ave_bias.append(ave)
    head = head - head_bias[index]
    return max_bias,ave_bias,min_bias




#bias_vec[0]->l bias_vec[1]->r
def main():
    paths='D:\\download\\new_data_framework\\s1\\R\\0\\25'
    plcr_=plcr()
    mode='l'
    get_calibration_parameter(mode,plcr_)
    test_one_points(paths,plcr_,24)




def test_one_position():
    path='D:\\download\\new_data_framework\\s1\\R\\2\\'
    bias_vec=np.array([-0.6671642, -0.83690633], dtype=np.float64).reshape((2, 1))
    none_vec=np.array([None])
    estimation=get_ave(path,bias_vec)
    global index
    index=2
    goal=get_goal()
    show_data(estimation,goal,8)

def test_one_points(r_path,plcr_,p_num):
    bias_vec=np.array([None])
    estimation=get_ave(r_path,plcr_)
    x=[]
    y=[]
    col=[]
    s=[]
    k=2
    goal=get_goal()
    for point in estimation:
        x.append(point[0][0])
        y.append(point[1][0])
        col.append(k)
        s.append(5)
        k+=1
    g=goal[p_num]
    x.append(g[0][0])
    y.append(g[1][0])
    col.append(k)
    s.append(10)
    plt.scatter(x,y,c=col,s=s)
    plt.show()


#mode:左 或者右边 es_module:plcr_
def test_one_point(path,es_module):
    num = []
    with open(path) as f:
        t = f.read()
        t = t.replace('\n', ' ')
        t = t.split()
        for i in t:
            num.append(trans(i))
        for i in range(len(t)):
            print(t[i], end=' ')
            if i & 1 == 1:
                print(' ')
    estimation = get_estimate(num,es_module=es_module)
    print(estimation)

def get_calibration_parameter(s,plcr_):
    if s=='l':
         plcr_.set_bias_vec(np.array([2.0487150869825754, -2.193507059394102], dtype=np.float64).reshape((2, 1)))
    elif s=='r':
        plcr_.set_calibration_angle(-math.sin(1.5*math.pi/180.0),-math.sin(1.5*math.pi/180.0))
    else:
        plcr_.set_bias_vec(np.array([None]))

if __name__=='__main__':
    main()

>>>>>>> master
