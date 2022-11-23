

a=0


class test_class(object):
    def __init__(self):
        pass
    def change_time(self):
        global a
        a+=10


def get_time():
    return a