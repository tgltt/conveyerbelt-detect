import os
import csv
import time

from PIL import Image
from collections import defaultdict

import torch



from utils import make_dir
from train_ssd300 import create_model
from mydataset import PREDICT_RESULT_TASK1
from mydataset import PREDICT_RESULT_TASK2
from mydataset import recover_task2_label
from mydataset import test_image_root
from transforms import Compose, Resize, ToTensor, Normalization

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

# 导出预测文件
def export_predict_result(image_root, train_weights, task1_file, task2_file):
    make_dir(path=task1_file)
    make_dir(path=task2_file)
    # 判断设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 构建模型
    # 目标检测数 + 背景
    num_classes = 2 + 1
    model = create_model(num_classes=num_classes)

    # 加载权重
    model.load_state_dict(torch.load(train_weights, map_location='cpu')['model'])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # initial model 到底是为什么？
        init_img = torch.zeros((1, 3, 600, 600), device=device)
        model(1, init_img)

        with open(file=task1_file, mode="w", encoding="utf8", newline='') as task1_csv_file:
            task1_header = ["filename", "label", "xmin", "ymin", "width", "height", "confidence"]
            task1_csv_writer = csv.DictWriter(f=task1_csv_file, fieldnames=task1_header)
            task1_csv_writer.writeheader()

            with open(file=task2_file, mode="w", encoding="utf8", newline='') as task2_csv_file:
                task2_header = ["filename", "label"]
                task2_csv_writer = csv.DictWriter(f=task2_csv_file, fieldnames=task2_header)
                task2_csv_writer.writeheader()

                for image_file_name in os.listdir(image_root):
                    # 读取图像
                    original_img = Image.open(os.path.join(image_root, image_file_name))

                    data_transform = Compose([Resize(),
                                              ToTensor(),
                                              Normalization()])
                    img, _ = data_transform(original_img)
                    # 改为批量预测
                    img = torch.unsqueeze(img, dim=0)

                    time_start = time_synchronized()
                    results = model(1, img.to(device))
                    time_end = time_synchronized()
                    print("{}, inference+NMS time: {}".format(image_file_name, time_end - time_start))

                    predictions = results[PREDICT_RESULT_TASK1][0]  # bboxes_out, labels_out, scores_out
                    predict_boxes = predictions[0].to("cpu").numpy()
                    if len(predict_boxes) > 0:
                        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
                        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]

                        predict_classes = predictions[1].to("cpu").numpy()
                        predict_scores = predictions[2].to("cpu").numpy()

                        for i in range(len(predict_boxes)):
                            if predict_scores[i] < 0.10:
                                continue

                            task1_new_row = defaultdict()

                            task1_new_row["filename"] = image_file_name
                            task1_new_row["label"] = predict_classes[i]

                            task1_new_row["xmin"] = int(predict_boxes[i, 0])
                            task1_new_row["ymin"] = int(predict_boxes[i, 1])
                            task1_new_row["width"] = int(predict_boxes[i, 2] - predict_boxes[i, 0])
                            task1_new_row["height"] = int(predict_boxes[i, 3] - predict_boxes[i, 1])

                            task1_new_row["confidence"] = round(predict_scores[i], 2)

                            task1_csv_writer.writerow(task1_new_row)

                            task2_result = results[PREDICT_RESULT_TASK2].argmax(dim=1)[0]

                            task2_new_row = defaultdict()
                            task2_new_row["filename"] = image_file_name
                            task2_new_row["label"] = recover_task2_label(task2_result.item())

                            task2_csv_writer.writerow(task2_new_row)

        print("导出 " + task1_file + " 成功")
        print("导出 " + task2_file + " 成功")
# 105_53.9_39.9
# 41_54.7_39.9 46_54.9_38.3

export_predict_result(image_root=test_image_root,
                      train_weights="./save_weights/server/ssd300-custom-pretrain-107.pth",
                      task1_file="submit/submit1.csv",
                      task2_file="submit/submit2.csv")