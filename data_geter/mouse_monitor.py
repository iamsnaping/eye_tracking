import PyHook3 as pyHook
import pythoncom
import pyautogui
i,j,k = 0,0,0

def funcLeft(event):
    if (event.MessageName != "mouse move"):  # 因为鼠标一动就会有很多mouse move，所以把这个过滤下
        global i
        i = i + 1
        print('第{:3d}次：按下鼠标左键我就会出现，嘻嘻'.format(i))
    return True

def funcMiddle(event):
    if (event.MessageName != "mouse move"):  # 因为鼠标一动就会有很多mouse move，所以把这个过滤下
        global j
        j = j + 1
        print('第{:3d}次：按下鼠标中键我就会出现，嘻嘻'.format(j))
    return True

def funcRight(event):
    if (event.MessageName != "mouse move"):  # 因为鼠标一动就会有很多mouse move，所以把这个过滤下
        global k
        k = k + 1
        print('第{:3d}次：按下鼠标右键我就会出现，嘻嘻'.format(k))
    return True

def main():
    hm = pyHook.HookManager()
    # 监听鼠标
    # hm.MouseLeftDown 是将“鼠标左键按下”这一事件和func这个函数绑定，即每次鼠标左键按下都会执行func
    # 如果希望监测鼠标中键按下则：hm.MouseMiddleDown，鼠标右键按下则：hm.MouseRightDown
    hm.MouseLeftDown = funcLeft  # 监测鼠标左键是否按下
    hm.MouseMiddleDown = funcMiddle  # 监测鼠标中键是否按下
    hm.MouseRightDown = funcRight  # 监测鼠标右键是否按下
    hm.HookMouse()

    # 循环监听
    pythoncom.PumpMessages()


if __name__ == '__main__':
    main()
