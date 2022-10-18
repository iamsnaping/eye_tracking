import sys

from pupil_detect.pupil_detect import *

from PyQt5 import QtWidgets

from PyQt5.QtWidgets import QWidget, QToolTip, QPushButton, QApplication, QLabel,QMainWindow, QHBoxLayout,QDesktopWidget,QButtonGroup,QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QFileDialog
import os
from test_.test import *


origin_path='C:\\Users\\snapping\\Desktop\\data\\origin'
process_path='C:\\Users\\snapping\\Desktop\\data\\process'






class data_show(QWidget):
    def __init__(self):
        super(data_show, self).__init__()
        self.button2=QPushButton('next')
        self.button1=QPushButton('last')
        self.bg=QButtonGroup(self)
        self.bg.addButton(self.button1,1)
        self.bg.addButton(self.button2,2)
        self.pic_id=0
        self.origin_path='C:\\Users\\snapping\\Desktop\\data\\origin'
        self.process_path='C:\\Users\\snapping\\Desktop\\data\\process'
        self.desktop=QApplication.desktop()
        self.w_height=int(self.desktop.height()*0.7)
        self.w_width=int(self.desktop.width()*0.35)
        self.origin_label=QLabel(self)
        self.process_label=QLabel(self)
        self.init_gui()

    def process_pic(self,img):
        width, length = img.shape
        for i in range(width):
            for j in range(width):
                img[i][j] = 0 if img[i][j] > 240 else 255
        return img


    def init_gui(self):
        y=int(self.desktop.height()*0.15)
        x=int(self.desktop.width()*0.075)
        self.setGeometry(x,y,self.w_width,self.w_height)
        width=int(self.desktop.width()*0.35/2)
        height=int(width*1080/1920)
        y=(self.w_height-height)/2
        self.origin_label.resize(width,height)
        self.process_label.resize(width,height)
        self.origin_label.move(0,y)
        self.process_label.move(width,y)
        self.origin_label.setScaledContents(True)
        self.origin_label.setToolTip('original')
        self.origin_label.setPixmap(QPixmap(os.path.join(origin_path,'0.png')))
        self.process_label.setPixmap(QPixmap(os.path.join(process_path,'0.png')))
        self.process_label.setScaledContents(True)
        self.process_label.setToolTip('processed')
        self.button1.move(width,height+y)
        bwidth=width*0.1
        bheight=height*0.1
        self.button1.setGeometry(width-bwidth,height+y,bwidth,bheight)
        self.button1.pressed.connect(self.click_button)
        self.button2.clicked.connect(self.click_button)
        self.button2.setGeometry(width,height+y,bwidth,bheight)
        # self.bg.addButton(self.button1,1)
        # self.bg.addButton(self.button2,2)
        # self.button1.setCheckable(True)
        self.button1.setParent(self)
        # self.button2.setCheckable(True)
        self.button2.setParent(self)
        self.process_label.setParent(self)
        self.origin_label.setParent(self)
        # self.bg.setParent(self)
        # self.bg.buttonPressed[int].connect(self.click_button)
        self.show()

    def resizeEvent(self, event):
        width = int(self.width() * 0.5)
        height = int(width * 1080 / 1920)
        print(self.width(),self.height())
        y = (self.height() - height) / 2
        bwidth = width * 0.1
        bheight = height * 0.1
        self.origin_label.resize(width, height)
        self.process_label.resize(width, height)
        self.origin_label.move(0, y)
        self.process_label.move(width, y)
        self.origin_label.setScaledContents(True)
        self.button1.setGeometry(width - bwidth, height + y, bwidth, bheight)
        self.button2.setGeometry(width, height + y, bwidth, bheight)

    def label_click(self):
        if self.origin_label == self.sender():
            self.pic_id -= 1
        elif self.process_label == self.sender():
            self.pic_id += 1
        # print(self.pic_id)
        if self.pic_id < 0:
            self.pic_id = 0
            message = '没有下一张了'
            QMessageBox.warning(self, 'warning', message)
        origin_path = os.path.join(self.origin_path, str(self.pic_id) + '.png')
        process_path = os.path.join(self.process_path, str(self.pic_id) + '.png')
        if os.path.exists(origin_path) and os.path.exists(process_path):
            self.process_label.setPixmap(QPixmap(process_path))
            self.origin_label.setPixmap(QPixmap(origin_path))
            return
        else:
            message = '没有上一张了'
        print('send')
        QMessageBox.warning(self, message)

    def click_button(self):
        if self.button1==self.sender():
            self.pic_id-=1
        elif self.button2==self.sender():
            self.pic_id+=1
        # print(self.pic_id)
        if self.pic_id<0:
            self.pic_id=0
            message='没有下一张了'
            QMessageBox.warning(self, 'warning',message)
        origin_path=os.path.join(self.origin_path,str(self.pic_id)+'.png')
        process_path=os.path.join(self.process_path,str(self.pic_id)+'.png')
        if os.path.exists(origin_path) and os.path.exists(process_path):
            img=get_ph(origin_path)
            img=self.process_pic(img)
            self.origin_label.setPixmap(QPixmap(origin_path))
            self.process_label.setPixmap(QPixmap())
            return
        else:
            message='没有上一张了'
        print('send')
        QMessageBox.warning(self,message)











if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = data_show()
    sys.exit(app.exec_())
