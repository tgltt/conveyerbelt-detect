# 计算训练集均值、标准差
import numpy as np
import cv2
import random

import os
from tqdm.notebook import tqdm_notebook

project_root = "D:\\workspace\\ai_study\\projects\\conveyer_belt_detector\\"
task2_train_classes_anno = os.path.join(project_root, "data\\task2\\train\\classes_all.txt")

# calculate means and std
train_txt_path = task2_train_classes_anno

CNum = 3500  # 挑选多少图片进行计算

img_h, img_w = 300, 300
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path, 'r') as f:
    lines = f.readlines()
    random.shuffle(lines)  # shuffle , 随机挑选图片

    for index in tqdm_notebook(range(CNum)):
        line = lines[index].strip().split(",")

        img = cv2.imread(line[0])
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32) / 255.

for i in tqdm_notebook(range(3)):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

# 输出：
# normMean = [0.3441988, 0.34242108, 0.3464927]
# normStd = [0.19682558, 0.19890308, 0.2000567]
# transforms.Normalize(normMean = [0.3441988, 0.34242108, 0.3464927], normStd = [0.19682558, 0.19890308, 0.2000567])
