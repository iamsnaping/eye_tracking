from base_estimation.plcr.plcr import plcr
import numpy as np

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


def plcr_main():
    t=plcr(34,27,43.4)
    t._rt=250
    t._radius=0.78
    t._pupil_center=np.array([255.961700, 287.527557,0]).reshape((3,1))
    t._param=np.array([0,0,0.42],dtype=np.float64).reshape((3,1))
    t.get_param()
    t._up=np.array([0,1,0],dtype=np.float64).reshape((3,1))
    light=np.array([294.485168 ,257.725494,0,250.682770 ,259.503143,0,251.993057 ,289.338593,0,293.811340, 288.928192,0],dtype=np.float64).reshape((4,3))
    light=light.T
    t._glints=t._pupil_center-light
    t._g0=np.array([272.941071 ,292.299927,0],dtype=np.float64).reshape((3,1))
    t._g0=t._pupil_center-t._g0
    t.get_e_coordinate()
    t.transform_e_to_i()
    t.get_plane()
    t.get_visual()
    t.get_m_points()
    print(t.gaze_estimation())


def plcr_main2():
    t=plcr(34,27,43.4)
    t._rt=250
    t._radius=0.78
    t._pupil_center=np.array([362.916962 ,220.244675,0]).reshape((3,1))
    t._param=np.array([0,0,0.42],dtype=np.float64).reshape((3,1))
    t.get_param()
    t._up=np.array([0,1,0],dtype=np.float64).reshape((3,1))
    light=np.array([364.757599 ,220.257584,0,320.601532 ,221.878540,0,321.264709, 253.619522,0,364.527313 ,251.839600,0],dtype=np.float64).reshape((4,3))
    light=light.T
    t._glints=t._pupil_center-light
    t._g0=np.array([342.520325 ,256.411835,0],dtype=np.float64).reshape((3,1))
    t._g0=t._pupil_center-t._g0
    t.get_e_coordinate()
    t.transform_e_to_i()
    t.get_plane()
    t.get_visual()
    t.get_m_points()
    print(t.gaze_estimation())

plcr_main()
plcr_main2()

