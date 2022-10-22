import cv2
from eye_utils import data_util as du
import numpy as np

img_path='C:\\Users\\snapping\\Desktop\\origin.jpg'
img=cv2.imread(img_path)
gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
print(gray_img.shape)
du.show_ph(gray_img)
# gaussian_img=cv2.GaussianBlur(gray_img,(3,3),0)
binary_img=cv2.threshold(gray_img,220,255,cv2.THRESH_BINARY)[1]
du.show_ph(binary_img)
canny=cv2.Canny(binary_img, 30, 60)
du.show_ph(canny)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 180)
print(lines)
# lines1 = lines[:, 0, :]
# for rho, theta in lines1[:]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a * rho
#     y0 = b * rho
#     x1 = int(x0 + 3000 * (-b))
#     y1 = int(y0 + 3000 * (a))
#     x2 = int(x0 - 3000 * (-b))
#     y2 = int(y0 - 3000 * (a))
#     cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

