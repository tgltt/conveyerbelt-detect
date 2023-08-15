from torch.utils.data import Dataset
import os
import torch
import csv
from collections import defaultdict
from PIL import Image

# 任务1标注文件字段索引
TASK1_CSV_FIELD_ID_INDEX = 0
TASK1_CSV_FIELD_FILENAME_INDEX = 1
TASK1_CSV_FIELD_LABEL_INDEX = 2
TASK1_CSV_FIELD_XMIN_INDEX = 3
TASK1_CSV_FIELD_YMIN_INDEX = 4
TASK1_CSV_FIELD_XMAX_INDEX = 5
TASK1_CSV_FIELD_YMAX_INDEX = 6
TASK1_CSV_FIELD_HEIGHT_INDEX = 7
TASK1_CSV_FIELD_WIDTH_INDEX = 8
TASK1_CSV_FIELD_CHANNEL_INDEX = 9
TASK1_CSV_FIELD_TASK_TYPE_INDEX=10
# 任务2标注文件字段索引
TASK2_CSV_FIELD_FILENAME_INDEX = 0
TASK2_CSV_FIELD_LABEL_INDEX = 1

TARGET_FIELD_TASK1_ANCHORS = "task1_anchors"
TARGET_FIELD_TASK1_LABELS = "task1_labels"
TARGET_FIELD_TASK2_LABEL = "task2_label"
TARGET_FIELD_IMAGE_ID = "image_id"
TARGET_FIELD_TASK1_AREA = "task1_area"
TARGET_FIELD_TASK1_ISCROWD = "task1_iscrowd"
TARGET_FIELD_HEIGHT_WIDTH = "height_width"
TARGET_FIELD_IMAGE_DATA = "image_data"
TARGET_FIELD_TASK_TYPE="task_type"

TASK_TYPE_ALL=1
TASK_TYPE_ONLY_TASK2=2

PREDICT_RESULT_TASK1 = "task1_result"
PREDICT_RESULT_TASK2 = "task2_result"

INVALID_FILE_INDEX = "-1"

IMAGE_SIZE = 300

TASK2_LABEL_COUNT = 2

# 数据预处理
project_root = "D:\\workspace\\ai_study\\projects\\conveyer_belt_detector"
dataset_root = "D:\\workspace\\ai_study\\dataset\\ConveyBeltAbnormalDetect\\dataset"

# 汇总的标注文件
task1_train_detect_anno = os.path.join(project_root, "data\\task1\\train\\train_info_all.csv")
task1_val_detect_anno = os.path.join(project_root, "data\\task1\\val\\val_infos.csv")
task2_train_classes_anno = os.path.join(project_root, "data\\task2\\train\\classes_all.txt")
task2_val_classes_anno = os.path.join(project_root, "data\\task2\\val\\classes.txt")

test_image_root = os.path.join(dataset_root, "test", "images")

pretrain_path = os.path.join(project_root, "res50+ssd" , "nvidia_ssdpyt_fp32_190826.pt")

def make_dir(path):
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        print("创建目录", dir_name)
        os.makedirs(dir_name)

make_dir(path=project_root)
make_dir(path=task1_train_detect_anno)
make_dir(path=task1_val_detect_anno)
make_dir(path=task2_train_classes_anno)
make_dir(path=task2_val_classes_anno)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_task2_label(label):
    return 1 if label == 3 else label


def recover_task2_label(label):
    return 3 if label == 1 else label


class MyDataset(Dataset):
    def __init__(self, task1_file="train_info_all.csv", task2_file="classes_all.txt", transforms=None,
                 val_data_flag=False):
        super(MyDataset, self).__init__()
        self.task1_file = task1_file
        self.task2_file = task2_file
        self.transforms = transforms

        self.images = []
        self.task1_anchors = []

        self.task1_labels = []
        self.task2_labels = []

        self.height_widths = []
        # 默认全流程, 后续根据标注文件进行修改
        self.task_types = []

        self.max_anchors_count_in_image = 0

        self.val_data_flag = val_data_flag

        self._read()

    def _read(self):
        def generate_empty_task2_labels_item():
            return {}

        task2_data = defaultdict(generate_empty_task2_labels_item)
        with open(file=self.task2_file, mode="r", encoding="utf8") as f:
            while True:
                line = f.readline().strip()
                if len(line) <= 0:
                    break

                line = line.split(",")

                task2_data[line[TASK2_CSV_FIELD_FILENAME_INDEX]] = int(line[TASK2_CSV_FIELD_LABEL_INDEX])

        with open(file=self.task1_file, mode="r", encoding="utf8") as f:
            line = f.readline()
            last_index = INVALID_FILE_INDEX
            anchors_count_in_image = 0
            while True:
                line = f.readline().strip()
                if len(line) <= 0:
                    break

                line = line.split(",")

                if line[TASK1_CSV_FIELD_ID_INDEX] != last_index:
                    if last_index != INVALID_FILE_INDEX:
                        if len(line) < 11 or int(line[10]) == 1:
                            self.images.append(last_file_name)
                            self.task1_anchors.append(task1_anchors)
                            self.task1_labels.append(task1_labels)
                            task2_label = task2_data[line[TASK1_CSV_FIELD_FILENAME_INDEX]]
                            self.task2_labels.append(transform_task2_label(label=task2_label))
                            self.height_widths.append(
                                [int(line[TASK1_CSV_FIELD_HEIGHT_INDEX]), int(line[TASK1_CSV_FIELD_WIDTH_INDEX])])

                            if not self.val_data_flag:
                                self.task_types.append(int(line[TASK1_CSV_FIELD_TASK_TYPE_INDEX]))
                        #                         print(task1_anchors)
                        #                         print(task1_labels)
                        #                         print(task2_data[line[TASK1_CSV_FIELD_FILENAME_INDEX]])
                        #                         print(self.height_widths)

                    task1_anchors = []
                    task1_labels = []

                    if self.max_anchors_count_in_image < anchors_count_in_image:
                        self.max_anchors_count_in_image = anchors_count_in_image

                    anchors_count_in_image = 0

                anchor_str = line[TASK1_CSV_FIELD_XMIN_INDEX:TASK1_CSV_FIELD_YMAX_INDEX + 1]
                task1_anchors.append([float(ele) for ele in anchor_str])
                task1_labels.append(int(line[TASK1_CSV_FIELD_LABEL_INDEX]))

                anchors_count_in_image += 1

                last_index = line[TASK1_CSV_FIELD_ID_INDEX]
                last_file_name = line[TASK1_CSV_FIELD_FILENAME_INDEX]

    def __getitem__(self, idx):
        image_id = torch.tensor([idx])
        # 图像数据
        image = self.images[idx]

        # BGR => RGB
        image = Image.open(fp=image).convert('RGB')

        # 真实框（标注结果）
        # (N, 4), N为标注框个数
        task1_anchors = []
        for anchor in self.task1_anchors[idx]:
            task1_anchors.append(torch.tensor(data=anchor).float())
        #         print(task1_anchors)

        # 任务一标签
        # (N), N为标注框个数
        task1_labels = []
        for label in self.task1_labels[idx]:
            # 标签是long类型
            task1_labels.append(torch.tensor(data=label).long())

        task1_anchors = torch.stack(tensors=task1_anchors, dim=0)
        task1_labels = torch.as_tensor(data=task1_labels, dtype=torch.int64)

        task2_label = torch.as_tensor(data=self.task2_labels[idx], dtype=torch.int64)
        image_id = torch.tensor(data=[image_id], dtype=torch.int64)
        height_width = torch.as_tensor(data=self.height_widths[idx], dtype=torch.int64)

        target = {}
        target[TARGET_FIELD_TASK1_ANCHORS] = task1_anchors.to(device=device)
        target[TARGET_FIELD_TASK1_LABELS] = task1_labels.to(device=device)
        target[TARGET_FIELD_TASK2_LABEL] = task2_label.to(device=device)
        target[TARGET_FIELD_IMAGE_ID] = image_id.to(device=device)
        target[TARGET_FIELD_HEIGHT_WIDTH] = height_width.to(device=device)

        if not self.val_data_flag:
            task_type = torch.as_tensor(self.task_types[idx], dtype=torch.int64)
            target[TARGET_FIELD_TASK_TYPE] = task_type.to(device=device)

        # 图像预处理
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # 返回结果
        return image, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = tuple(zip(*batch))
        return images, targets

    def coco_index(self, idx):
        """
        该方法是专门为 pycocotools 统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        boxes = []
        labels = []

        boxes.extend(self.task1_anchors[idx])
        labels.extend(self.task1_labels[idx])
        iscrowds = [0 for _ in range(len(boxes))]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64)
        height_width = torch.as_tensor(self.height_widths[idx], dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target[TARGET_FIELD_TASK1_ANCHORS] = boxes.to(device=device)
        target[TARGET_FIELD_TASK1_LABELS] = labels.to(device=device)
        target[TARGET_FIELD_IMAGE_ID] = image_id.to(device=device)
        target[TARGET_FIELD_TASK1_AREA] = area.to(device=device)
        target[TARGET_FIELD_TASK1_ISCROWD] = iscrowds.to(device=device)
        target[TARGET_FIELD_HEIGHT_WIDTH] = height_width.to(device=device)

        return target