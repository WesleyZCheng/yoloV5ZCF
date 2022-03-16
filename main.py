# -*- coding: UTF-8 -*-
import os
import numpy as np
import k_means_m
# import pandas as pd
#import matplotlib as mpl
# import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import time
import json
from PIL import Image
import cv2
import colorsys
import random
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def out_json(index, image, Inf_rice_list, Location_rice_list):
    image_save = 'D:/fei/pycharm代码/Non_destructive_testing/inference/image_save/'
    image_label = 'D:/fei\pycharm代码/Non_destructive_testing/inference/image_label/'
    image_imshow = np.copy(image)
    width, height, _ = image.shape
    labels = np.array(Inf_rice_list)[:, 0]
    # Location_rice = np.array(Location_rice_list)
    dict_json = {}
    out_contour_rice = []
    out_contour_broken = []
    out_images = np.zeros((width, height, len(Location_rice_list)), dtype=np.float32)
    for j, Location_rice in enumerate(Location_rice_list):
        out_image = np.zeros((width, height), dtype=np.float32)
        for i in range(Location_rice.shape[0]):
            # out_image[Location_rice[i, 0], Location_rice[i, 1]] = 1.
            out_images[Location_rice[i, 0], Location_rice[i, 1], j] = 1.
        # contours = measure.find_contours(out_image, 1)
        out_image = out_images[:, :, j].astype(np.uint8)
        contours, hierarchy = cv2.findContours(out_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # row = np.where(hierarchy[0, :, 2] == -1)[1][0]
        num = 0
        if len(contours) > 1:
            length_contour = 0
            for i, contour in enumerate(contours):
                if contour.shape[0] > length_contour:
                    length_contour = contour.shape[0]
                    num = i
        if labels[j] == 0:
            out_contour_rice.append(contours[num].squeeze(1).tolist())
        if labels[j] == 1:
            out_contour_broken.append(contours[num].squeeze(1).tolist())
    # out_image = morphology.remove_small_objects(out_image, min_size=1000, connectivity=1, in_place=False)
    #显示实例分割结果
    color = random_colors(len(Location_rice_list))  # 随机生成颜色
    alpha = 0.5
    for i in range(out_images.shape[-1]):
        for c in range(3):
            image_imshow[:, :, c] = np.where(out_images[:, :, i] == 1,
                                            image_imshow[:, :, c] *
                                            (1 - alpha) + alpha * color[i][c] * 255,
                                            image_imshow[:, :, c])
    plt.imsave('D:/fei/pycharm代码/Non_destructive_testing/inference/image_imshow/imshow_' + str(index) + '.jpg', image_imshow)
    out_images = np.sum(out_images, axis=-1)
    out_images = np.expand_dims(out_images, axis=-1).repeat(3, axis=-1) * image
    out_images = out_images.astype(np.uint8)
    # plt.figure()
    # plt.imshow(out_images)
    # plt.show()
    plt.imsave(image_save + str(index) + '.jpg', out_images)

    dict_json['rice'] = out_contour_rice
    dict_json['broken_rice'] = out_contour_broken

    filename = image_label + str(index) + '.json'
    with open(filename, 'w') as f:
        json.dump(dict_json, f, indent=2, separators=(',', ': '))
    out_contour_rice.extend(out_contour_broken)

    # plt.figure()
    # for i in out_cont  our_rice:
    #     i = np.array(i)
    #     plt.scatter(i[:, 0], i[:, 1])
    # plt.show()

    return 0

def ConnectedArea(BW):
    #标记连通区域
    BW = clear_border(BW)
    labels = measure.label(BW, connectivity=1)
    #去除噪声点，噪声连通区域像素点个数小于min_size connectivity=1表示四连通，2表示八连通 in_place=False表示直接在输入图像中删除小块区域
    img1 = morphology.remove_small_objects(labels, min_size=1000, connectivity=1, in_place=False)
    # plt.imshow(img1, cmap='binary')
    # plt.show()
    #获取各连通域属性
    label_att = measure.regionprops(img1)
    return label_att

def rgb2bw(*args):
    image = args[0]
    image_gray = color.rgb2gray(image)
    if len(args) < 2:
        threshold = filters.threshold_otsu(image_gray)
    else:
        threshold = args[1]
    image_bw = image_gray >= threshold
    if len(args) < 2:
        return image_bw, threshold
    else:
        return image_bw

def adhersion_image_extract(index, data_image, data_txt, interval):
    i = 0
    data_txt[:, [1, 2, 3, 4]] = data_txt[:, [2, 1, 4, 3]]   #行列对调
    while i < data_txt.shape[0]:
        temp1 = np.setdiff1d(list(range(0, data_txt.shape[0])), i)
        if temp1.size != 0:
            temp2 = data_txt[i, 1:3] - data_txt[temp1, 1:3]#去重框
            distance_temp = min(data_txt[i, 3] - data_txt[i, 1], data_txt[i, 4] - data_txt[i, 2])/3
            if temp2.size != 0 and min(np.linalg.norm(temp2, axis=1)) < distance_temp:
                data_txt = np.delete(data_txt, i, axis=0)
                i -= 1
            i += 1
        else:
            i += 1
    grain_center = (data_txt[:, (3, 4)] + data_txt[:, (1, 2)])/2
    grain_gap = data_txt[:, (3, 4)] - data_txt[:, (1, 2)]
    image_bw, _ = rgb2bw(data_image)
    label_att = ConnectedArea(image_bw)
    if len(label_att)==0:
        return None, None
    centroid = np.zeros((len(label_att), 2))
    for i in range(0, len(label_att)):
        centroid[i, :] = label_att[i].centroid
    if centroid.shape[0] <= grain_center.shape[0]:      #centroid保存连通域质心 grian_center保存框中心
        label_connected = []
        for i in grain_center:
            row_temp = np.argmin(np.linalg.norm(i-centroid, axis=1), axis=0)    #将框中心与连通域中心对应
            label_connected.append(row_temp)
    else:
        print('error:框的数量小于连通域个数')
    # 矩阵连接inf_grain:所有框的整米碎米标签（0或1），框的连通域编号，23列为框的中心位置,45列为框左上角坐标，67列为框右下角坐标，89列为框大小
    # if len(label_connected)==0:
    #     label_connected = list(-1 * np.ones(data_txt.shape[0]))
    inf_grain = np.concatenate((np.stack((data_txt[:, 0], label_connected), axis=1), grain_center, data_txt[:, 1:5], grain_gap), axis=1)
    # 如果存在粘连域
    row_inf_ad_temp = []
    inf_adhersion = []
    Location_adhersion = []
    row_inf_rice = range(0, inf_grain.shape[0])
    # time_start = time.time()
    if len(label_connected) - len(np.unique(label_connected)) > 0:
    #找到粘连域编号
        label_repeat = [val for val in set(label_connected) if label_connected.count(val) >= 2]
        row_inf_adhersion = []
        row_inf_ad = []
        for i in label_repeat:
            row_inf_ad_temp = []
            for ii in range(0, len(label_connected)):
                if label_connected[ii] == i:
                    row_inf_ad_temp.append(ii)
                    row_inf_ad.append(ii)
            row_inf_adhersion.extend(row_inf_ad_temp)
            min_temp = [min(label_att[i].coords[:, 0]), min(label_att[i].coords[:, 1])]         #min_temp储存该粘连域横纵坐标最小值 用于裁剪粘连域，减少k均值计算量
            Label_temp = label_att[i].coords - min_temp
            adhersion_bw = np.zeros(np.max(Label_temp, axis=0)+1)
            for j in Label_temp:
                adhersion_bw[j[0], j[1]] = 1
            inf_adhersion_temp = inf_grain[row_inf_ad_temp]
            pixel_classed = image_classify(adhersion_bw, interval, inf_adhersion_temp[:, 2:4]-min_temp)
            if np.isnan(pixel_classed).sum():
                print('error:image_classify函数返回labels错误')
            pixel_classed[:, (0, 1)] = pixel_classed[:, (0, 1)] + min_temp
            # time21 = time.time()
            for j in range(0, inf_adhersion_temp.shape[0]):
                row_temp = np.where(pixel_classed[:, 2]==j)[0]
                if row_temp.size:
                    Location_ad_temp = pixel_classed[row_temp, :]
                    Location_adhersion.append(Location_ad_temp[:, (0, 1)])
                    inf_adhersion.append(inf_adhersion_temp[j])
            # time2 = time.time()
            # print('time2 - time21', time2 - time21)
            # print('time2 - time1', time2 - time1)
        row_inf_rice = np.setdiff1d(row_inf_rice, row_inf_ad)
    # time_end = time.time()
    # print('time_end - time_start', time_end - time_start)
    inf_rice = list(inf_grain[row_inf_rice])
    Location_rice = []
    for i in range(0, len(row_inf_rice)):
       Location_rice.append(label_att[int(inf_rice[i][1])].coords)
    for j in range(0, len(inf_adhersion)):
        Location_rice.append(Location_adhersion[j])
        inf_rice.append(inf_adhersion[j])

    out_json(index, data_image, inf_rice, Location_rice)
    # color = random_colors(len(Location_rice))  # 随机生成颜色
    ###############画出每张图片使用k_means聚类后的效果图（未去重）#######################################
    # plt.figure()
    # for j in range(0, len(inf_rice)):
    #     plt.scatter(Location_rice[j][:, 0], Location_rice[j][:, 1], color=color[j])
    # plt.axis('off')
    # plt.savefig('D:/fei\pycharm代码/Non_destructive_testing/inference/image_imshow/' + str(index) + '.jpg', dpi=500, bbox_inches = 'tight')
    # plt.show()
    ###############################################################################################
    a = np.array(inf_rice)
    b = sum(a[:, 0])
    print('\n')
    print('rice_num', a.shape[0])
    print('broken_num', b)
    return (inf_rice, Location_rice)

def image_classify(BW, interval, center):
    #BW：二值化图像，interval：间隔
    #返回原图像素位置信息与标签
    pixel_classed = []
    if interval < 1 or interval > min(BW.shape):
        print('error: interval invalid')
    k = 0
    image_all = np.zeros((BW.shape[0], BW.shape[1], pow(interval, 2)))
    for i in range(0, interval):
        temp_x = np.setdiff1d(range(0, BW.shape[0]), range(i, BW.shape[0], interval))
        image_temp_x = np.copy(BW)
        image_temp_x[temp_x, :] = 0
        for j in range(0, interval):
            temp_y = np.setdiff1d(range(0, BW.shape[1]), range(j, BW.shape[1], interval))
            image_temp_y = np.copy(image_temp_x)
            image_temp_y[:, temp_y] = 0
            image_all[:, :, k] = image_temp_y
            pos1 = np.where(image_temp_y > 0)
            k = k+1
            labels = k_means_m.kmeans_m(np.array(pos1), np.transpose(center))
            pixel_classed_temp = np.concatenate((np.transpose(pos1), labels), axis=1)
            for ii in pixel_classed_temp:
                pixel_classed.append(ii)
    return np.array(pixel_classed)

def txt_read(file_Path):
    data_txt = []
    with open(file_Path, 'r') as txt_to_read:
        while True:
            lines = txt_to_read.readline()
            if not lines:
                break
            p_tmp = [float(i) for i in lines.split()]
            data_txt.append(p_tmp)
    return np.array(data_txt)

def remove_repetition(inf_rice1, inf_rice2, Location_rice2, delta):
    record_i = []
    record_match_num = []
    inf_rice1_numpy = np.concatenate((np.array(inf_rice1), np.ones((len(inf_rice1), 1))*(-1), np.zeros((len(inf_rice1), 1))), axis=1)
    inf_rice2_numpy = np.array(inf_rice2)
    for i in range(0, len(inf_rice1)):
        row_match = []
        distance = abs(inf_rice1_numpy[i, 3] - inf_rice2_numpy[:, 3])
        IX = np.argsort(distance)
        B = np.sort(distance)
        row_distance_min = np.where(B < 200)
        if row_distance_min[0].size == 0:
            continue
        temp1 = abs(inf_rice1_numpy[i, 8] - inf_rice2_numpy[IX[row_distance_min], 8])       #两颗米粒列差
        temp2 = abs(inf_rice1_numpy[i, 9] - inf_rice2_numpy[IX[row_distance_min], 9])       #两颗米粒行差
        row_temp_min = np.where((temp1 < 50) & (temp2 < 50))
        if row_temp_min[0].size == 1:
            inf_rice1_numpy[i, 10] = IX[row_temp_min[0]]
            inf_rice1_numpy[i, 11] = inf_rice1_numpy[i, 2] - inf_rice2_numpy[IX[row_temp_min[0]], 2]
        if row_temp_min[0].size > 1:
            record_i.append(i)
            record_match_num.append(IX[row_temp_min])
    # delta = np.median(inf_rice1_numpy[np.where(inf_rice1_numpy[:, 10]>=0), 11])
    # print(delta)
    bound = [delta-100, delta+100]
    for i in range(0, len(record_i)):
        j = record_i[i]
        match_num = record_match_num[i]
        distance_delta = abs(abs(inf_rice1_numpy[j, 2] - inf_rice2_numpy[match_num, 2]) - delta)
        inf_rice1_numpy[j, 10] = match_num[np.argmin(distance_delta)]
        inf_rice1_numpy[j, 11] = inf_rice1_numpy[j, 2] - inf_rice2_numpy[match_num[np.argmin(distance_delta)], 2]
    a = np.where(inf_rice1_numpy[:, 10] >= 0)[0]            #找到所有匹配到的行
    b = np.where((abs(inf_rice1_numpy[a, 11]) < bound[0]) | (abs(inf_rice1_numpy[a, 11]) > bound[1]))[0]      #只删除匹配行中坐标范围内的行
    if b.size < a.size:
        delta_num = inf_rice1_numpy[np.setdiff1d(a, a[b]), 10]
        delta_num_int = np.sort(np.unique(delta_num.astype(int)))           #排序去除相同的删除序号
        inf_rice2_numpy = np.delete(inf_rice2_numpy, delta_num_int, axis=0)
        i = delta_num_int.size - 1
        while i >= 0:
            Location_rice2.pop(delta_num_int[i])
            i -= 1

    ##########################画出去掉重复米粒后的剩下米粒分类图#############################################################
    # plt.figure(figsize=(11, 11), dpi=100)
    # for i in range(0, len(Location_rice2)):
    #     temp = Location_rice2[i]
    #     plt.scatter(temp[:, 1], 2160-temp[:, 0])
    # plt.axis('off')
    # plt.show()                        #必须人工叉掉显示出的图片  程序才会继续向下执行
    ####################################################################################################################
    print('去重后--rice_num:', inf_rice2_numpy.shape[0])
    print('去重后--broken_num:', sum(inf_rice2_numpy[:, 0]))
    # plt.axis('off')
    # plt.show()
    # plt.pause(1)
    # plt.close()
    return (inf_rice2_numpy, Location_rice2)

def main(delta):
    start_index = 1

    interval = 5
    filePath_image = 'D:/fei\pycharm代码/Non_destructive_testing/inference/images/'
    filePath_txt = 'D:/fei/pycharm代码/Non_destructive_testing/inference/label/'
    name = os.listdir(filePath_image)
    files = len(name)
    data_image1 = io.imread(filePath_image + str(start_index) + '.jpg')
    data_txt1 = txt_read(filePath_txt + str(start_index) + '.txt')
    inf_rice1, Location_rice1 = adhersion_image_extract(1, data_image1, data_txt1, interval)
    inf_rice_all = []
    Location_rice_all = []
    inf_rice_all.append(np.array(inf_rice1))
    Location_rice_all.append(np.array(Location_rice1, dtype=object))
    for i in range(start_index + 1, files + 1):
        print('循环次数：', i)
        data_image2 = io.imread(filePath_image + str(i) + '.jpg')
        data_txt2 = txt_read(filePath_txt + str(i) + '.txt')
        time1 = time.time()
        inf_rice2, Location_rice2 = adhersion_image_extract(i, data_image2, data_txt2, interval)
        time2 = time.time()
        print('adhersion_image_extract spend time', time2 - time1)
        if inf_rice1 is not None and inf_rice2 is not None:
            inf_rice2, Location_rice2 = remove_repetition(inf_rice1, inf_rice2, Location_rice2, delta)
        time3 = time.time()
        print('remove_repetition spend time', time3 - time2)
        inf_rice_all.append(np.array(inf_rice2))
        Location_rice_all.append(np.array(Location_rice2, dtype=object))
        inf_rice1 = inf_rice2
    return inf_rice_all, Location_rice_all

if __name__ == '__main__':
    delta = 262   #fps = 20 262  FPS=30 121                        #####每次单独运行main 时 必须根据pre-processing获得的delta_main替换该值###########
    time1 = time.time()
    inf_rice_all, Location_rice_all = main(delta)
    time2 = time.time()
    print('all_time', time2 - time1)
    broken_rice_num = 0
    rice_num = 0
    for i in range(0, len(inf_rice_all)):
        inf_rice = np.array(inf_rice_all[i])
        broken_rice_num += sum(inf_rice[:, 0])
        rice_num += inf_rice.shape[0]
    print('大米总数', rice_num)
    print('碎米总数', broken_rice_num)
