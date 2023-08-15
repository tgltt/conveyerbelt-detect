# 预测
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time

import torch
from PIL import Image
import matplotlib.pyplot as plt

from ssd_model import Backbone, SSD300
from transforms import Compose, Resize, ToTensor, Normalization
from mydataset import PREDICT_RESULT_TASK1
from draw_utils import draw_objs
from utils import make_dir

dataset_root = "D:\\workspace\\ai_study\\dataset\\ConveyBeltAbnormalDetect\\dataset"

def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def predict(image_root, image_file_name, output_dir):
    # 判断设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 构建模型
    # 目标检测数 + 背景
    num_classes = 2 + 1
    model = create_model(num_classes=num_classes)

    # 加载权重
    train_weights = "./save_weights/server/ssd300-88.pth"
    model.load_state_dict(torch.load(train_weights, map_location='cpu')['model'])
    model.to(device)

    # 读取图像
    original_img = Image.open(os.path.join(image_root, image_file_name))

    data_transform = Compose([Resize(),
                              ToTensor(),
                              Normalization()])
    img, _ = data_transform(original_img)
    # 改为批量预测
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # initial model 到底是为什么？
        init_img = torch.zeros((1, 3, 300, 300), device=device)
        model(1, init_img)

        time_start = time_synchronized()
        predictions = model(1, img.to(device))[PREDICT_RESULT_TASK1][0]  # bboxes_out, labels_out, scores_out
        time_end = time_synchronized()
        print("inference+NMS time: {}".format(time_end - time_start))
        predict_boxes = predictions[0].to("cpu").numpy()
        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
        predict_classes = predictions[1].to("cpu").numpy()
        predict_scores = predictions[2].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index={"1": "Metal", "2": "Sundries"},
                             box_thresh=0.5,
                             line_thickness=8,
                             font='arial.ttf',
                             font_size=30)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        result_file = os.path.join(output_dir, image_file_name)
        make_dir(path=result_file)
        plot_img.save(result_file)


class1_val_image_root = os.path.join(dataset_root, "class1", "val")
class2_val_image_root = os.path.join(dataset_root, "class2", "val")
class3_val_image_root = os.path.join(dataset_root, "class3", "val")
normal_val_image_root = os.path.join(dataset_root, "normal", "val")
test_image_root = os.path.join(dataset_root, "test", "images")

class1_val_image_output_root = os.path.join(class1_val_image_root, "predict")
class2_val_image_output_root = os.path.join(class2_val_image_root, "predict")
class3_val_image_output_root = os.path.join(class3_val_image_root, "predict")
test_image_output_root = os.path.join(test_image_root, "predict")

# images = os.listdir(class1_val_image_root)
# for image in images:
#     predict(image_root=class1_val_image_root, image_file_name=image, output_dir=class1_val_image_output_root)

# images = os.listdir(class2_val_image_root)
# for image in images:
#     predict(image_root=class2_val_image_root, image_file_name=image, output_dir=class2_val_image_output_root)

# images = os.listdir(test_image_root)
# for image in images:
#     predict(image_root=test_image_root, image_file_name=image, output_dir=test_image_output_root)
