import math

import numpy as np
import os
import cv2
from eye_utils import data_util as du

root_path = 'C:\\Users\\snapping\\Desktop\\data\\2022.10.9\\left\\'
pic_path = os.path.join(root_path, '8.png')

origin_img = cv2.imread(pic_path)
gray_img = du.get_gray_pic(origin_img)
gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 1)
# for i in range(200):
#     for j in range(200):
#         gaussian_img[i][j]=0 if (gaussian_img[i][j]<70 or gaussian_img[i][j]>=150)else np.uint8(255)
# du.show_ph(gaussian_img)
canny_img = cv2.Canny(gaussian_img, 30, 40)


# shun shi zhen kai tou shi yao qu diao de

def img_filter(img, filters, threshold=0, degrade=0, fill_up=0):
    filter_len = len(filters[0])
    for i in range(200):
        for j in range(200):
            if img[i][j]!=255:
                continue
            for filter in filters:
                flag = 0
                for k in range(0, filter_len):
                    a = i + filter[k][0]
                    b = j + filter[k][1]
                    # print(filter[k])
                    if not (0 <= a < 200) or not (0 <= b < 200):
                        break
                    if img[a][b]!=255:
                        break
                    flag+=int(img[a][b])
                if flag == threshold:
                    if degrade != 0:
                        for k in range(0, degrade):
                            a = i + filter[k][0]
                            b = j + filter[k][1]
                            img[a][b] = np.uint8(0)
                    if fill_up != 0:
                        for k in range(filter_len - fill_up, filter_len):
                            a = i + filter[k][0]
                            b = j + filter[k][1]
                            img[a][b] = np.uint8(255)
    return img


def img_filters(img, filters, threshold=None, degrade=None, fill_up=None, filter_len=None):
    assert degrade is not None
    assert fill_up is not None
    assert filter_len is not None
    if threshold is None:
        threshold = []
        for i, j in zip(fill_up, filter_len):
            threshold.append((j-i) * 255)
    filters_len = len(filters)
    for i in range(200):
        for j in range(200):
            if img[i][j]!=255:
                continue
            for l in range(filters_len):
                for filter in filters[l]:
                    flag = 0
                    for k in range(0, filter_len[l]):
                        a = i + filter[k][0]
                        b = j + filter[k][1]
                        if (not (0 <= a < 200)) or (not (0 <= b < 200)):
                            break
                        if img[a][b]!=255:
                            break
                        flag += int(img[a][b])
                    if flag == threshold[l]:
                        if degrade != 0:
                            for k in range(0, degrade[l]):
                                a = i + filter[k][0]
                                b = j + filter[k][1]
                                img[a][b] = np.uint8(0)
                        if fill_up != 0:
                            for k in range(filter_len[l] - fill_up[l], filter_len[l]):
                                a = i + filter[k][0]
                                b = j + filter[k][1]
                                img[a][b] = np.uint8(255)
    return img


filter_1 = [[[0, 0],[1,0],[0,-1]],
            [[0, 0],[-1,0],[0,1]],
            [[0, 0],[0,1],[1,0] ],
            [[0, 0], [0,-1],[-1,0]]]
filter_2 = [[[0, 0], [-1, 1], [1, 1], [0, 1]],
            [[0, 0], [1, 1], [1, -1], [1, 0]],
            [[0, 0], [1, -1], [-1, -1], [0, -1]],
            [[0, 0], [-1, -1], [-1, 1], [-1, 0]]]

filter_3 = [[[0, 0], [0, 1], [1, 2], [1, -1], [1, 0], [1, 1]],
            [[0, 0], [-1, 0], [-2, 1], [1, 1], [0, 1], [-1, 1]],
            [[0, 0], [1, 0], [2, -1], [-1, -1], [0, -1], [1, -1]],
            [[0, 0], [0, -1], [-1, -2], [-1, 1], [-1, 0], [-1, -1]]]
filter_4 = [[[0, 0], [1, 1], [2, 1], [0, -1]],
            [[0, 0], [-1, -1], [-2, -1], [0, 1]],
            [[0, 0], [0, 1], [1, -1], [2, -1]],
            [[0, 0], [0, -1], [-2, 1], [-1, 1]]]
filter_5 = [[[0, 0], [1, 1], [2, 1], [3, 1], [-1, -3], [-1, -2], [-1, -1]],
            [[0, 0], [-1, -1], [-2, -1], [-3, -1], [1, 1], [1, 2], [1, 3]],
            [[0, 0], [-1, 1], [-1, 2], [-1, 3], [3, -1], [2, -1], [1, -1]],
            [[0, 0], [1, -1], [1, -2], [1, -3], [-3, 1], [-2, 1], [-1, 1]]]
filter_6 = [[[0, 0], [1, 1], [2, 2], [2, -2], [1, -1]],
            [[0, 0], [-1, 1], [-2, 2], [2, 2], [1, 1]],
            [[0, 0], [1, -1], [2, -2], [-2, -2], [-1, -1]],
            [[0, 0], [-1, -1], [-1, -1], [-2, 2], [-1, 1]]]
filter_7 = [[[0, 0], [1, 1], [2, 2], [2, -3], [1, -2], [0, -1]],
            [[0, 0], [-1, 1], [-2, 2], [3, 2], [2, 1], [1, 0]],
            [[0, 0], [1, -1], [2, -2], [-3, -2], [-2, -1], [-1, 0]],
            [[0, 0], [-1, -1], [-2, -2], [-2, 3], [-1, 2], [0, 1]]]


du.show_ph(canny_img)
filter_img=canny_img.copy()
# filter_img=img_filter(filter_img,filter_2,threshold=255*3,degrade=1,fill_up=1)
filter_img = img_filters(filter_img, [filter_2, filter_3], degrade=[1, 2], fill_up=[1, 2], filter_len=[4, 6])
# filter_img = img_filter(canny_img, filter_1, threshold=255 * 3, degrade=1)
# filter_img = img_filters(filter_img, [filter_4, filter_5, filter_6, filter_7], degrade=[1, 1, 1, 1],
#                          fill_up=[0, 0, 0, 0], filter_len=[4, 7, 5, 6])

# for i in range(200):
#     for j in range(200):
#         if canny_img[i][j]!=0:
#             for filter in filter_1:
#                 flag=0
#                 for k in range(3):
#                     a=i+filter[k][0]
#                     b=j+filter[k][1]
#                     if not (a>0 and a<200) or not (a>0 and a<200):
#                         break
#                     flag+=int(filter_img[a][b])
#                 if flag==255*3:
#                     filter_img[i+filter[0][0]][j+filter[0][1]]=0
#             for filter in filter_2:
#                 for k in range(4):
#                     a=i+filter[k][0]
#                     b=i+filter[k][1]
#                     if not (a>0 and a<200) or not (a>0 and a<200):
#                         break
# canny_img=filter_img.copy()
# filter_img=canny_img.copy()
# for i in range(200):
#     for j in range(200):
#         if canny_img[i][j]!=0:
#             for filter in filter_2:
#                 flag=0
#                 for k in range(3):
#                     a=i+filter[k][0]
#                     b=j+filter[k][1]
#                     if not (a > 0 and a < 200) or not (a > 0 and a < 200):
#                         break
#                     flag+=int(filter_img[a][b])
#                 if flag==255*3:
#                     filter_img[i+filter[0][0]][j+filter[0][1]]=0
#                     filter_img[i+filter[3][0]][j+filter[3][1]]=255
#
#

du.show_ph(filter_img,'filtered')
canny_img = filter_img
contours_old, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
contours=[]
glints_contours=[]
pupil_contours=[]
c_len=len(contours_old)
for i in range(c_len):
    pass
    if hierarchy[0][i][3]==-1:
        if hierarchy[0][i][2]!=-1:
            contours.append(contours_old[i])
            contours.append(contours_old[hierarchy[0][i][2]])
        #     contours.append(np.concatenate((contours_old[i],contours_old[hierarchy[0][i][2]])))
        # else:
        #     contours.append(contours_old[i])
print(c_len)
print(len(contours))
# contours_old=contours.copy()
contours=[]
for contour in contours_old:
    if len(contour)<5:
        continue

    ellipse = cv2.fitEllipseAMS(contour)
    if math.isnan(ellipse[1][0]) or math.isnan(ellipse[1][1]):
        continue
    rth = ellipse[1][0] / ellipse[1][1]
    if rth < 0.5:
        continue
    if ellipse[1][1] < 5 or ellipse[1][0] > 60:
        continue
    if (ellipse[1][0]>10 and ellipse[1][1] < 30):
        continue
    if not (0<=ellipse[0][0]<200) or not (0<=ellipse[0][1]<200):
        continue
    if ellipse[1][0] >10 and ellipse[1][1] <30:
        continue
    if ellipse[1][0]<10:
        glints_contours.append(contour)
    if ellipse[1][0]>30:
        pupil_contours.append(contour)
    # print(rth)
    contours.append(contour)
# contours=pupil_contours

print(len(contours))
# contours[8]=np.concatenate((contours[8],contours[11]))
contours=tuple(contours)
img = np.zeros_like(canny_img, dtype=np.uint8)
drawed_img = cv2.drawContours(img, contours, -1, (255, 255, 255))

contours_num=[i for i in range(len(contours))]
# contours_num=[0,1,2,3,4,5]
drawed_img = cv2.cvtColor(drawed_img, cv2.COLOR_GRAY2RGB)
for i in contours_num:
    # ellipse = cv2.fitEllipseDirect(contours[i])
    # center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
    # axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
    # drawed_img = cv2.ellipse(origin_img, [center, axes, ellipse[2]], color=(0, 0,255))
    # print(f'direct red ellipse {ellipse}')
    # ellipse = cv2.fitEllipse(contours[i])
    # center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
    # axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
    # drawed_img = cv2.ellipse(drawed_img, [center, axes, ellipse[2]], color=(255, 0, 0))
    # print(f'fit blue ellipse {ellipse}')
    ellipse = cv2.fitEllipseAMS(contours[i])
    center = (math.ceil(ellipse[0][0]), math.ceil(ellipse[0][1]))
    axes = (math.ceil(ellipse[1][0]), math.ceil(ellipse[1][1]))
    drawed_img = cv2.ellipse(drawed_img, [center, axes, ellipse[2]], color=(0, 255, 0))
    print(f'ams green ellipse{ellipse}')




cv2.namedWindow('d', 0)
cv2.resizeWindow('d', 200, 200)
cv2.imshow('d', drawed_img)
cv2.waitKey()
