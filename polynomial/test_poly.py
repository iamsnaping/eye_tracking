import np
import numpy as np
import os
import matplotlib.pyplot as plt
import random
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from polynomial.pr_cr.pc_cr import pc_cr
import test_main as file_util
import matplotlib.pyplot as plt
def get_num(s):
    s=s.replace('\n',' ')
    s=s.split(' ')
    num=file_util.get_num(s)
    return num

def get_one_position(p_path):
    list_dir=os.listdir(p_path)
    nums=[]
    for dir in list_dir:
        p=os.path.join(p_path,dir)
        p_dirs=os.listdir(p)
        for ff in p_dirs:
            p_f=os.path.join(p,ff)
            with open(p_f) as f:
                t = f.read()
                nums.append(get_num(t))
            f.close()
    return nums


def get_one_eye_data(p_path):
    list_dir=os.listdir(p_path)
    nums=[]
    for dirs in list_dir:
        p=os.path.join(p_path,dirs)
        c_dirs=os.listdir(p)
        for dir in c_dirs:
            c_p=os.path.join(p,dir)
            dd=os.listdir(c_p)
            for d in dd:
                ff=os.path.join(c_p,d)
                with open(ff) as f:
                    t=f.read()
                    nums.append(get_num(t))
                f.close()
    return nums


def test_one_opsition(num_p):
    t_pc_cr = pc_cr(16)
    goals=file_util.get_goal()

    data_p = get_one_position(num_p)
    # print(f'total_num {total_num}')
    total_num = len(data_p)
    cali_nums = []
    print(total_num)
    for i in range(16):
        cali_nums.append(random.randint(0, total_num))
        # cali_nums.append(i*20+1)
        # cali_nums.append(i)
    cali_data = [data_p[i] for i in cali_nums]
    calibration = []
    goal = []
    for i in cali_nums:
        goal.append(goals[((i % (20 * 49) // 20) - int((i % 20) == 0))])
        # print(f'this is i {i} point {((i%(20*49)//20)-int((i%20)==0))} point_ {i%(20*49)}')
    for cali in cali_data:
        calibration.append(t_pc_cr.get_vector(
            [np.array([cali[10], cali[11]]).reshape((2, 1)), np.array([cali[6], cali[7]]).reshape((2, 1)),
             np.array([cali[8], cali[9]]).reshape((2, 1))]))

    t_pc_cr.do_calibration(calibration, goal)
    head = np.array([17.0, 13.5, 57.5])
    i = 0
    ave_angle = [0.0 for i in range(49)]
    x_ = []
    y_ = []
    col = []
    for point in data_p:
        estimation = t_pc_cr.do_estimation(
            [np.array([point[10], point[11]]).reshape((2, 1)), np.array([point[4], point[5]]).reshape((2, 1)),
             np.array([point[2], point[3]]).reshape((2, 1))])
        v1 = np.array([estimation[0, 0], estimation[1, 0], 0]) - head
        x_.append(estimation[0,0])
        y_.append(estimation[1,0])
        col.append((i//20)+1)
        v1 /= np.linalg.norm(v1)
        v2 = np.array([goals[i // 20][0][0], goals[i // 20][1][0],0]) - head
        v2 /= np.linalg.norm(v2)
        alpha = np.acos(np.dot(v1, v2)) *180.0/np.pi
        ave_angle[i // 20] += alpha
        i += 1
    for i in range(49):
        ave_angle[i] /= 20.0
    plt.scatter(x_,y_,c=col,s=5)
    plt.show()
    # i = 0
    # for cali in cali_data:
    #     print(t_pc_cr.do_estimation(
    #         [np.array([cali[10], cali[11]]).reshape((2, 1)), np.array([cali[4], cali[5]]).reshape((2, 1)),
    #          np.array([cali[2], cali[3]]).reshape((2, 1))]), end=' ')
    #     print(goal[i])
    #     i += 1
def test_one_eye(p_path):
    t_pc_cr = pc_cr(10)
    goals = file_util.get_goal()
    cali_nums = []
    data=get_one_position(p_path)
    total_num=len(data)
    for i in range(10):
        cali_nums.append(random.randint(0, total_num))
        # cali_nums.append(i)
    cali_data = [data[i] for i in cali_nums]
    calibration = []
    goal = []
    for i in cali_nums:
        goal.append(goals[((i % (20 * 49) // 20) - int((i % 20) == 0))])
    for cali in cali_data:
        calibration.append(t_pc_cr.get_vector(
            [np.array([cali[10], cali[11]]).reshape((2, 1)), np.array([cali[4], cali[5]]).reshape((2, 1)),
             np.array([cali[2], cali[3]]).reshape((2, 1))]))

    t_pc_cr.do_calibration(calibration, goal)
    head=np.array([17.5,13.5,57.5])
    i=0
    ave_angle=[0.0 for i in range(49)]
    for point in data:
        estimation=t_pc_cr.do_estimation([np.array([point[10], point[11]]).reshape((2,1)),np.array([point[4],point[5]]).reshape((2,1)),
                                              np.array([point[2],point[3]]).reshape((2,1))])
        v1=np.array([estimation[0,0],estimation[1,0],0])-head
        v1/=np.linalg.norm(v1)
        v2=np.array([goals[i//20,0],estimation[i//20,1],0])-head
        v2/=np.linalg.norm(v2)
        alpha=np.acos(np.dot(v1,v2))*np.pi/180.0
        ave_angle[i//20]+=alpha
        i+=1
    for i in range(49):
        ave_angle[i]/=20.0
    print(ave_angle)

    i = 0
    # for cali in cali_data:
    #     print(t_pc_cr.do_estimation(
    #         [np.array([cali[10], cali[11]]).reshape((2, 1)), np.array([cali[4], cali[5]]).reshape((2, 1)),
    #          np.array([cali[2], cali[3]]).reshape((2, 1))]), end=' ')
    #     print(goal[i])
    #     i += 1

def main():
    p='D:\\download\\new_data_framework\\s1\\L'
    num_p='D:\\download\\new_data_framework\\s1\\L\\0'
    test_one_opsition(num_p)


if __name__=='__main__':
    main()
