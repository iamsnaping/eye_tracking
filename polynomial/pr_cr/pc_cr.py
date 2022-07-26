from polynomial.base_polynomial import base_polynomial

class pc_cr(base_polynomial):
    def __int__(self,calib_num,order):
        super(pc_cr, self).__int__(calibration_num=calib_num)
        self._order=order
        # self._metric=
    def get_matric(self,x,y):
        pass
    def do_estimation(self,point):
        pass