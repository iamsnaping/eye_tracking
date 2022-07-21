from plcr.plcr import  plcr
from crdd.crdd import crdd
import numpy as np

# 3,3,0          -2,2,0
#          0,0,0
# 2,-3,0       -1,-2,0

def plcr_main():
    t=plcr(30,40,50)
    t._param=np.array([1,1,0.5],dtype=np.float64).reshape((3,1))
    t.get_param()
    up=np.array([1,2,3,2,3,4,3,4,5,4,5,6],dtype=np.float64).reshape((4,3))
    t._up=np.array([0,1,0],dtype=np.float64).reshape((3,1))
    light=np.array([-0.2,0.2,0,0.3,0.3,0,0.2,-0.3,0,-0.1,-0.2,0],dtype=np.float64).reshape((4,3))
    light=light.T
    t._glints=light
    t._g0=np.array([0,-0.1,0],dtype=np.float64).reshape((3,1))
    t.get_e_coordinate()
    t.get_plane()
    t.get_visual()
    t.get_m_points()
    print(t.gaze_estimation())


#plcr_main()

t=crdd(30,40,50)
