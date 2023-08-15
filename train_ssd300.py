import os
import datetime

import torch

from torch.utils.data import DataLoader

from PIL import ImageFile

from ssd_model import Backbone, SSD300
from transforms import *
from mydataset import MyDataset
from mydataset import task1_train_detect_anno, task2_train_classes_anno
from mydataset import task1_val_detect_anno, task2_val_classes_anno
from coco_utils import get_coco_api_from_dataset
from train_eval_utils import train_one_epoch
from train_eval_utils import evaluate
from plot_curv import plot_map, plot_loss_and_lr

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_model(num_classes=2 + 1):
    # 先构建一个 backbone
    backbone = Backbone()

    # 再构建一个SSD300
    model = SSD300(backbone=backbone, num_classes=num_classes)

    pre_ssd_path = "res50+ssd\\pretrain-ssd300-54.pth"
    if not os.path.exists(pre_ssd_path):
        raise FileNotFoundError("nvidia_ssdpyt_fp32.pt not find in {}".format(pre_ssd_path))
    pre_model_dict = torch.load(pre_ssd_path, map_location='cpu')
    pre_weights_dict = pre_model_dict["model"]

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model


def main(parser_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 保存训练完成之后的权重
    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    # 定义一个结果文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    """
        1，数据读取后的处理工作
            - 类型转换
            - 数据增强
    """

    data_transform = {
        "train": Compose([SSDCropping(),  # 图像切割
                          Resize(),  # 统一大小
                          ColorJitter(),  # 颜色抖动
                          ToTensor(),  # 转张量
                          RandomHorizontalFlip(),  # 水平翻转
                          Normalization(),  # 标准化
                          AssignGTtoDefaultBox()]),  # 处理目标框和锚框

        "val": Compose([Resize(),
                        ToTensor(),
                        Normalization()])
    }

    # 定义训练集
    train_dataset = MyDataset(task1_file=task1_train_detect_anno,
                              task2_file=task2_train_classes_anno,
                              transforms=data_transform["train"])

    # 注意训练时，batch_size必须大于1
    batch_size = parser_data.batch_size

    assert batch_size > 1, "batch size must be greater than 1"

    # 防止最后一个batch_size=1，如果最后一个batch_size=1就舍去
    drop_last = True if len(train_dataset) % batch_size == 1 else False

    # 数据预处理多少线程
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   # num_workers=nw,
                                   collate_fn=train_dataset.collate_fn,
                                   drop_last=drop_last)

    # 定义验证集
    val_dataset = MyDataset(task1_file=task1_val_detect_anno,
                            task2_file=task2_val_classes_anno,
                            transforms=data_transform["val"],
                            val_data_flag=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  # num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)
    # 定义模型
    model = create_model(num_classes=args.num_classes + 1)
    model.to(device)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params=params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   step_size=10,
                                                   gamma=0.8)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    #     optimizer.param_groups[0]["lr"] = 0.00001

    train_loss = []
    learning_rate = []
    val_map = []

    # 提前加载验证集数据，以免每次验证时都要重新加载一次数据，节省时间
    val_data = get_coco_api_from_dataset(val_data_loader.dataset)

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        mean_loss, lr = train_one_epoch(model=model, optimizer=optimizer,
                                        data_loader=train_data_loader,
                                        device=device, epoch=epoch,
                                        print_freq=10)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        print(f"Epoch:{epoch} finish, mean_loss:{mean_loss}, lr:{lr}")

        # 更新学习率
        lr_scheduler.step()

        # 测试数据
        coco_info = evaluate(epoch=epoch, model=model, data_loader=val_data_loader,
                             device=device, data_set=val_data)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/ssd300-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    # 检测的目标类别个数，不包括背景(替换：自己的检测类别)
    parser.add_argument('--num_classes', default=2, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=5, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args(args=[])

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
