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


def get_data(root_path,pic_path_name='origin'):
        list_dir = os.listdir(root_path)
        pic_path=os.path.join(root_path,pic_path_name)
        if not os.path.exists(pic_path):
            os.mkdir(pic_path)
        t = 0
        k = -1
        for v_list in list_dir:
            if os.path.splitext(v_list)[1] != '.avi':
                continue
            v_path = os.path.join(root_path, v_list)
            cap = cv2.VideoCapture(v_path)
            while (cap.isOpened()):
                ref, frame = cap.read()
                k += 1
                if k % 5 != 0:
                    continue
                if ref:
                    img_path = os.path.join(pic_path, str(t) + '.png')
                    cv2.imwrite(img_path, frame)
                    t += 1
                else:
                    break
