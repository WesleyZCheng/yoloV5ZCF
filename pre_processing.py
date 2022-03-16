import imageio
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage import measure, morphology
import main
import numpy as np
import time
import argparse
import cv2
import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *
from detect import detect

#每次运行程序前需要删除文件夹里面的文件:D:\fei\pycharm代码\Non_destructive_testing\inference 删除该路径下三个子文件中的照片和txt文件（文件夹勿删）
#拍摄的视频需要转成.avi格式的视频  视频中皮带输送机方向必须是上下方向，视频帧大小一般是2160*3840
#单独运行main模块时 必须将delta——main的值 替换main中246行的delta值，，一般可以不动
from utils.general import check_img_size


def remove_repetition2(label_att1, label_att2):
    a = []
    att1 = []
    att2 = []
    for i in label_att1:
        att1.append(np.concatenate((np.array(i.centroid), [i.area, i. bbox_area, i.perimeter]), axis=0))
    for i in label_att2:
        att2.append(np.concatenate((np.array(i.centroid), [i.area, i. bbox_area, i.perimeter]), axis=0))
    inf_rice1 = np.array(att1)           #[centroid, area, bbox_area, perimeter]
    inf_rice2 = np.array(att2)           #[centroid, area, bbox_area, perimeter]
    delta = []
    for i in range(0, len(att1)):
        distance = abs(inf_rice1[i, 1] - inf_rice2[:, 1])############################################################
        IX = np.argsort(distance)
        B = np.sort(distance)
        row_distance_min = np.where(B < 30)[0]
        if row_distance_min.shape[0] == 0:
            continue
        temp1 = abs(inf_rice1[i, 2] - inf_rice2[IX[row_distance_min], 2])
        temp2 = abs(inf_rice1[i, 3] - inf_rice2[IX[row_distance_min], 3])
        temp3 = abs(inf_rice1[i, 4] - inf_rice2[IX[row_distance_min], 4])
        row_temp_min = np.where((temp1 < 2000) & (temp2 < 2000) & (temp3 < 50))
        if len(row_temp_min[0]) != 1:
            continue
        else:
            delta.append(inf_rice1[i, 0] - inf_rice2[IX[row_temp_min[0]], 0])
    if delta is not None:
        delta = np.array(delta)
        delta_median = np.median(delta)
    else:
        delta_median =[]
    return delta_median


if __name__ == '__main__':
    time_start = time.time()
    filePath_image = 'D:/fei/pycharm代码/Non_destructive_testing/inference/images/'
    cap = cv2.VideoCapture('D:/fei/Grain_video_capture_images/20201118Rice/VID_20201118_14.avi')

    interval = 40               #计算fps时的人工取帧间隔
    image = []
    frame_num = cap.get(7)          #获得总帧数
    frame_count = int(frame_num/2)
    frame_count_delta = 0
    image_height, image_width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    while cap.isOpened() and frame_count_delta <= interval * 5:
        if frame_count_delta % interval == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            image.append(frame)
        frame_count_delta += interval
        frame_count += interval
    delta = []
    i = 0
    thresholds = []
    while i < 5:
        image_bw1, threshold_temp = main.rgb2bw(image[i])
        thresholds.append(threshold_temp)
        label_att1 = main.ConnectedArea(image_bw1)
        image_bw2, threshold_temp = main.rgb2bw(image[i+1])
        thresholds.append(threshold_temp)
        label_att2 = main.ConnectedArea(image_bw2)
        delta.append(remove_repetition2(label_att1, label_att2))
        i += 1
    threshold = np.mean(thresholds)
    frame_count = 0
    delta_stand = abs(np.median(np.array(delta)) / interval)
    fps = int((image_width*(4/5)) / delta_stand)####################################################

    fps = 20############################################################################################################################################################################################################

    delta_main = fps * delta_stand            #取帧后 两帧之间的偏移量
    i = 1
    while frame_count <= frame_num:
        if frame_count % fps == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, im = cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            BW = main.rgb2bw(im, threshold)
            BW = clear_border(BW)
            # labels = measure.label(BW, connectivity=1)
            # img1 = morphology.remove_small_objects(labels, min_size=1000, connectivity=1, in_place=False)            # 去除噪声点，噪声连通区域像素点个数小于min_size connectivity=1表示四连通，2表示八连通 in_place=False表示直接在输入图像中删除小块区域
            for j in range(0, im.shape[2]):  # 去掉背景
                im[:, :, j] = BW * im[:, :, j]
            imageio.imsave(filePath_image + str(i) + '.jpg', im)
            print(i)
            i += 1
        frame_count += fps
    cap.release()

    time1 = time.time()
    print('pre_processing time_cost : ', time1 - time_start)
## 深度学习

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/整米碎米--2000（无粘连）/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect(opt=opt)


    time2 = time.time()
    print('deep learning time_cost :   ', time2 - time1)
    inf_rice_all, Location_rice_all = main.main(delta_main)
    broken_rice_num = 0
    rice_num = 0
    for inf_rice in inf_rice_all:
        try:
            broken_rice_num += sum(inf_rice[:, 0])
            rice_num += inf_rice.shape[0]
        except IndexError:
            continue
    time_end = time.time()
    print('pre_processing time_cost : ', time1 - time_start)
    print('deep learning time_cost :   ', time2 - time1)
    print('main time_cost : ', time_end - time2)
    print('all time_cost : ', time_end - time_start)
    print('broken_rice_num : ', broken_rice_num)
    print('rice_num : ', rice_num)
    pass
