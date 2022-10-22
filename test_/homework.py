import cv2
import math
import numpy as np


class point3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def show_img(name, img):
    width = img.shape[1]
    length = img.shape[0]
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, width, length)
    cv2.imshow(name, img)
    cv2.waitKey()


def GetLineFromPoints(point1, point2, lines):
    l1 = (point1.y * point2.z - point1.z * point2.y)
    l2 = (point1.z * point2.x - point1.x * point2.z)
    l3 = (point1.x * point2.y - point1.y * point2.x)
    l1 = l1 / l3
    l2 = l2 / l3
    l3 = 1.0
    lines.append([l1,l2,l3])


def ComputeEquationSet(coefficients, result):
    a0 = coefficients[0][0]
    b0 = coefficients[0][1]
    c0 = coefficients[0][2]
    a1 = coefficients[1][0]
    b1 = coefficients[1][1]
    c1 = coefficients[1][2]
    s12 = (a0 * c1 - a1 * c0) / (a1 * b0 - a0 * b1)
    s11 = -(b0 * s12 + c0) / a0
    s22 = 1.0
    result.append(s11)
    result.append(s12)
    result.append(s22)


def ComputeCholesky(s, K):
    s11 = s[0]
    s12 = s[1]
    x11 = math.sqrt(s11)
    x21 = s12 / s11 * x11
    x22 = math.sqrt(1 - x21 * x21)
    x11 = x11 / x22
    x21 = x21 / x22
    x22 = 1.0
    K.append(x11)
    K.append(0.0)
    K.append(x21)
    K.append(x22)


def GetRectifingImage1(H, src, dst):
    height, width = src.shape
    for i in range(height):
        for j in range(width):
            x3 = 1.0
            x1 = round(j * H[0, 0] / x3)
            x2 = round((j * H[1, 0] + i) / x3)
            if 0 <= x1 < width and 0 <= x2 < height:
                dst[x2, x1] = np.uint8(src[i, j])
    show_img("src1", src)
    show_img("dst", dst)


def RectifyByOrthogonal(points, src):
    lines = []
    num_lines = 4
    for i in range(num_lines):
        GetLineFromPoints(points[2 * i], points[2 * i + 1], lines)
    coefficients = []
    for i in range(2):
        coefficient = []
        coefficient.append(lines[i * 2][0] * lines[i * 2 + 1][0])
        coefficient.append(lines[i * 2][0] * lines[i * 2 + 1][1] + lines[i * 2][1] * lines[i * 2 + 1][0])
        coefficient.append(lines[i * 2][1] * lines[i * 2 + 1][1])
        coefficients.append(coefficient)
    s = []
    ComputeEquationSet(coefficients, s)
    K = []
    ComputeCholesky(s, K)
    Ha = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Ha[0, 0] = 1 / K[0]
    Ha[0, 1] = 0
    Ha[1, 0] = -K[2] / K[0]
    Ha[1, 1] = 1
    img = np.zeros_like(src)
    GetRectifingImage1(Ha, src, img)


def RectifyAffine():
    img_path = "C:\\Users\\snapping\\Desktop\\rectify\\picture.png"
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    points_3d_8 = [point3d(23, 71, 1), point3d(101, 126, 1), point3d(101, 126, 1), point3d(238, 76, 1),
                   point3d(50, 91, 1), point3d(196, 91, 1), point3d(141, 59, 1), point3d(101, 126, 1)]
    RectifyByOrthogonal(points_3d_8, src)
    cv2.fitLine()


if __name__ == '__main__':
    RectifyAffine()
