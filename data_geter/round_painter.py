import random
from tkinter import *


TRANSCOLOUR = 'gray'

tk = Tk()
tk.geometry('500x400+500+150')
tk.title('test')
tk.wm_attributes('-transparentcolor', TRANSCOLOUR)
    # tk.bind('<Configure>',on_resize)
canvas = Canvas(tk)
canvas.pack(fill=BOTH, expand=Y)
tk.update()


def create_ball(canvas,tk):
    # tkinter绘图采用屏幕坐标系，原点在左上角，x从左往右递增，y从上往下递增
    # 在绘图区域内，随机产生当前球的圆心的x坐标和y坐标，用于制定出现的位置
    xpos = random.randint(10, int(tk.winfo_width())-10)
    ypos = random.randint(10, int(tk.winfo_height())-10)

    # 随机产生表示当前球的大小，也就是半径长度
    radius = 10

    # 通过lambda表达式创建函数对象r，每次调用r()都会产生0~255之间的数字
    r = lambda: random.randint(0, 255)

    # 三次调用的数字取前两位，用十六进制数方式存储到self.color里，作为球的颜色
    # RRGGBB，前2是红色，中2是绿色，后2是蓝色，最小是0，最大是F

    # 如全黑#000000  全白#FFFFFF  全红#FF0000
    color = "#%02x%02x%02x" % (255, 0, 0)

    # canvas.create_oval可以绘制一个圆
    # 但是需要传入圆的左、上、右、下四个坐标
    # 所以我们先产生4个坐标，通过这个四个坐标，绘制圆的大小

    # 左坐标=x坐标-半径
    x1 = xpos - radius
    # 上坐标=y坐标-半径
    y1 = ypos - radius
    # 右坐标=x坐标+半径
    x2 = xpos + radius
    # 下坐标=y坐标+半径
    y2 = ypos + radius

    # 通过canvas.create_oval()方法绘出整个圆，填充色和轮廓色分别是self.color生成的颜色
    canvas.delete('ball')
    ball = canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color,tag='ball')
    # canvas.addtag_all('t5')
    # print(x1,y1,x2,y2,color,radius)

    canvas.after(2000, create_ball,canvas,tk)

def on_resize(evt):
    tk.configure(width=evt.width,height=evt.height)
    canvas.create_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), fill=TRANSCOLOUR, outline=TRANSCOLOUR)
    print(canvas.winfo_width())



def main():
    create_ball(canvas, tk)
    tk.mainloop()


if __name__=='__main__':
   main()
