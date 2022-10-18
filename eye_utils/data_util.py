import cv2
import os



def get_gray_pic(img):
    return cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)

def show_ph(img, name='img', wait_time=False):
    width = img.shape[1]
    length = img.shape[0]
    cv2.namedWindow(name, 0)
    cv2.resizeWindow(name, width, length)
    cv2.imshow(name, img)
    if wait_time == False:
        cv2.waitKey()
    elif type(wait_time)==int:
        cv2.waitKey(wait_time)


def get_video(root_path, filename_extension='.avi'):
    f_list = os.listdir(root_path)
    video_name = []
    for f in f_list:
        if os.path.splitext(f) != filename_extension:
            continue
        video_name.append(os.path.join(root_path, f))
    return video_name
