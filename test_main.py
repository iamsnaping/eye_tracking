from base_estimation.plcr.plcr import plcr
import numpy as np

# 342.520325 256.411835
# 320.601532 221.878540
# 364.757599 220.257584
# 364.527313 251.839600
# 321.264709 253.619522
# 362.916962 220.244675 64.084343 74.064552 136.148544


def plcr_main():
    t=plcr(34,27,43.4)
    t._rt=100
    t._radius=0.78
    t._pupil_center=np.array([337.967194 ,265.266083,0]).reshape((3,1))
    t._param=np.array([0,0,0.42],dtype=np.float64).reshape((3,1))
    t.get_param()
    t._up=np.array([0,1,0],dtype=np.float64).reshape((3,1))
    light=np.array([345.340942 ,245.991333,0,301.462463 ,247.462662,0,303.031555 ,277.640015,0,345.104431 ,277.099243,0],dtype=np.float64).reshape((4,3))
    light=light.T
    t._glints=t._pupil_center-light
    t._g0=np.array([323.500763 ,280.579742,0],dtype=np.float64).reshape((3,1))
    t._g0=t._pupil_center-t._g0
    t.get_e_coordinate()
    t.transform_e_to_i()
    t.get_plane()
    t.get_visual()
    t.get_m_points()
    print(t.gaze_estimation())


plcr_main()

