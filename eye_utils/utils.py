import numpy as np
import torch


def get_points(l1,l2):
    l1[0]=l1[0].reshape((2,1))
    l1[1] = l1[1].reshape((2, 1))
    l2[0] = l2[0].reshape((2, 1))
    l2[1] = l2[1].reshape((2, 1))

    x0,x1,y0,y1=l1[0][0][0],l1[1][0][0],l1[0][1][0],l1[1][1][0]
    x2,x3,y2,y3=l2[0][0][0],l2[1][0][0],l2[0][1][0],l2[1][1][0]
    points=torch.zeros((2,1),dtype=np.float64)
    points[0][0]=((y1-y0)*x1*(x2-x3)-(y2-y3)*(x1-x0)*x2+(y2-y1)*(x1-x0)*
                  (x2-x3))/((y1-y0)*(x2-x3)-(y2-y3)*(x1-x0))
    points[1][0]=((y0-y1)*(points[0][0]-x1)+y1*(x0-x1))/(x0-x1)
    return points




def get_points_3d(*kwargs):
    p0=kwargs[0].reshape((1,3))
    p1=kwargs[1].reshape((1,3))
    p2=kwargs[2].reshape((1,3))
    p3=kwargs[3].reshape((1,3))
    v1=(p0-p1)/np.linalg.norm(p0-p1)
    v2=(p3-p2)/np.linalg.norm(p3-p2)
    t=np.linalg.norm(np.cross((p2-p1),v2))/np.linalg.norm(np.cross(v1,v2))
    point=p1+t*v1
    return point

