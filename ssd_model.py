import torch
from torch import nn, Tensor
from torch.jit.annotations import List

import random

from resnet50 import resnet50

from mydataset import PREDICT_RESULT_TASK1
from mydataset import PREDICT_RESULT_TASK2

from utils import dboxes300_coco, Encoder, PostProcess
from utils import device

from draw_utils import save_anchor_pic

class Backbone(nn.Module):
    """
        定义一个backbone
    """

    def __init__(self, pretrain_path=None):
        super(Backbone, self).__init__()

        # 定义一个resnet50
        net = resnet50()

        # 后续的通道数
        self.out_channels = [2048, 1024, 1024, 512, 512, 512]

        # 加载预训练模型
        if pretrain_path is not None:
            net.load_state_dict(torch.load(pretrain_path))

        # 截取特征提出部分
        self.feature_extractor = nn.Sequential(*list(net.children())[:7])

        # 修改其中的属性
        conv4_block1 = self.feature_extractor[-1][0]
        # 修改conv4_block1的stride，从2->1
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


# 主模型
class SSD300(nn.Module):
    """
        SSD300主模型
    """

    def __init__(self, backbone=None, num_classes=21):
        """
            设置分类的数量

        """
        super(SSD300, self).__init__()

        # 参数校验
        if backbone is None:
            raise Exception("backbone is None")

        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")

        self.feature_extractor = backbone

        self.num_classes = num_classes

        # [b, 1024, 76, 76] -> [b, 2048, 38, 38]
        self.conv = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2)

        # 构建自定义的特征层
        # out_channels = [2048, 1024, 1024, 512, 512, 512] for resnet50
        self._build_additional_features(self.feature_extractor.out_channels)

        # 每个特征层上每个特征点对应的锚框数量
        self.num_defaults = [4, 6, 6, 6, 4, 4]

        self.belt_pos = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, padding=0)

        location_extractors = []
        confidence_extractors = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        # 模型锚框的生成策略
        default_box = dboxes300_coco()

        # 下面三个是核心
        self.compute_loss = Loss(default_box)
        self.encoder = Encoder(default_box)
        self.postprocess = PostProcess(default_box)

    def _build_additional_features(self, input_size):
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size:
        :return:
        """
        additional_blocks = []
        # input_size = [2048, 1024, 1024, 512, 512, 512] for resnet50
        # shape :          76   [38,   19,   10,   5,   3]
        # input_size[:-1]：1024 [2048, 1024, 1024, 512, 512]
        # middle_channels：     [512,  512,  256,  256, 256]
        # input_size[1:]：      [1024, 1024, 512,  512, 512]
        middle_channels = [512, 512, 256, 256, 256]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            padding, stride = (1, 2) if i < 3 else (0, 1)
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=padding, stride=stride, bias=False),
                nn.BatchNorm2d(output_ch),
                #                 nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)

        self.additional_blocks = nn.ModuleList(additional_blocks)

        shortcut_blocks = [
            # [b, 1024, 76, 76] -> [b, 1024, 19, 19]
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=4),
            # [b, 2048, 38, 38] -> [b, 1024, 10, 10]
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=4),
            # [b, 1024, 19, 19] -> [b, 512, 5, 5]
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=4),
            # [b, 1024, 10, 10] -> [b, 512, 3, 3]
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=4),
            # [b, 512, 5, 5] -> [b, 512, 3, 3]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=2)
        ]

        self.shortcut_blocks = nn.ModuleList(shortcut_blocks)

    def _init_weights(self):
        """
            要不要自己去初始化权重？？？
        """
        layers = [self.conv, *self.additional_blocks, *self.shortcut_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, eopch, image, targets=None):
        # [b, 3, 600, 600] -> [b, 1024, 76, 76]
        x = self.feature_extractor(image)

        shortcuts = []
        shortcuts.append(self.shortcut_blocks[0](x))

        # [b, 1024, 76, 76] -> [b, 2048, 38, 38]
        x = self.conv(x)

        # Feature Map 2048, 19x19x1024, 10x10x1024, 5x5x512, 3x3x512, 1x1x512
        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
        detection_features.append(x)

        relu_inplace = nn.ReLU(inplace=True)
        relu = nn.ReLU(inplace=False)

        additional_blocks_count = len(self.additional_blocks)
        for i in range(additional_blocks_count):
            # for sub_layer in layer.children():
            #    try:
            #        print(f"{sub_layer}, weight.grad:{sub_layer.weight.grad}")
            #    except:
            #        pass

            # shape          :  76   [38,   19,   10,   5,   3]
            # input_size[:-1]： 1024 [2048, 1024, 1024, 512, 512]
            # middle_channels：      [512,  512,  256,  256, 256]
            # input_size[1:]：       [1024, 1024, 512,  512, 512]
            if i < 4:
                shortcuts.append(self.shortcut_blocks[i + 1](x))

            if i >= 1:
                x = self.additional_blocks[i](relu_inplace(x + shortcuts[i - 1]))
                if i == additional_blocks_count - 1:
                    x = relu_inplace(x)
            else:
                x = relu_inplace(self.additional_blocks[i](x))

            detection_features.append(x)

        # [b, 512, 1, 1] -> [b, 2, 1, 1]
        ptask2_label = self.belt_pos(x)
        # [b, 2, 1, 1] -> [b, 2]
        ptask2_label = ptask2_label.reshape(-1, 2)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        # 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            # print(bboxes_out.is_contiguous())
            labels_out = targets['labels']
            # print(labels_out.is_contiguous())

            #             if epoch % 20 == 0:
            #                 self.show_train_effect(image, locs, confs, bboxes_out, labels_out)

            # ploc, plabel, gloc, glabel
            loss = self.compute_loss(locs,
                                     confs,
                                     bboxes_out,
                                     labels_out,
                                     ptask2_label,
                                     targets['task2_label'])
            return {"total_losses": loss}

        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = {}
        results[PREDICT_RESULT_TASK1] = self.postprocess(locs, confs)
        results[PREDICT_RESULT_TASK2] = ptask2_label
        return results

    def show_train_effect(self, imgs, locs_pred, confs_pred, bboxes_out, labels_out):

        batch_size = locs_pred.shape[0]
        # 从当前批次任选一图片，展示预测的偏移准确度效果
        choice = random.randint(0, batch_size - 1)
        # [32, 4, 8732], [32, 3, 8732] -> [4, 8732], [3, 8732]
        locs_pred, confs_pred = locs_pred[choice], confs_pred[choice]
        # [32, 4, 8732], [32, 8732] -> [4, 8732], [8732]
        bboxes_out, labels_out = bboxes_out[choice], labels_out[choice]

        # [4, 8732] -> [8732, 4]
        locs_pred = locs_pred.permute(1, 0)
        # [3, 8732] -> [8732, 3]
        confs_pred = confs_pred.permute(1, 0)
        # [4, 8732] -> [8732, 4]
        bboxes_out = bboxes_out.permute(1, 0)

        # [8732] -> [8732]
        true_flag = labels_out > 0

        # [8732, 4] -> [N', 4]
        locs_pred = locs_pred[true_flag]
        # [8732, 4] -> [N']
        confs_pred = confs_pred[true_flag]

        pred_scores, confs_pred = torch.max(input=confs_pred, dim=1)
        # [8732, 4] -> [N', 4]
        bboxes_out = bboxes_out[true_flag]
        # [8732] -> [N']
        labels_out = labels_out[true_flag]

        bboxes = torch.zeros_like(bboxes_out).to(device=device)
        bboxes[:, 0] = bboxes_out[:, 0] - bboxes_out[:, 2] * 0.5
        bboxes[:, 1] = bboxes_out[:, 1] - bboxes_out[:, 3] * 0.5
        bboxes[:, 2] = bboxes_out[:, 0] + bboxes_out[:, 2] * 0.5
        bboxes[:, 3] = bboxes_out[:, 1] + bboxes_out[:, 3] * 0.5

        dboxes_xywh = dboxes300_coco()(order='xywh').to(device=device)
        dboxes_xywh = dboxes_xywh[true_flag]
        locs_pred[:, :2] = 0.1 * locs_pred[:, :2]  # 预测的x, y回归参数
        locs_pred[:, 2:] = 0.2 * locs_pred[:, 2:]  # 预测的w, h回归参数

        # 将预测的回归参数叠加到default box上得到最终的预测边界框
        locs_pred[:, :2] = locs_pred[:, :2] * dboxes_xywh[:, 2:] + dboxes_xywh[:, :2]
        locs_pred[:, 2:] = locs_pred[:, 2:].exp() * dboxes_xywh[:, 2:]

        # transform format to ltrb
        l = locs_pred[:, 0] - 0.5 * locs_pred[:, 2]
        t = locs_pred[:, 1] - 0.5 * locs_pred[:, 3]
        r = locs_pred[:, 0] + 0.5 * locs_pred[:, 2]
        b = locs_pred[:, 1] + 0.5 * locs_pred[:, 3]

        locs_pred[:, 0] = l  # xmin
        locs_pred[:, 1] = t  # ymin
        locs_pred[:, 2] = r  # xmax
        locs_pred[:, 3] = b  # ymax

        #         print(f"train: bboxes:{bboxes[:5].detach()}, labels_out:{labels_out[:5].detach()}")
        #         print(f"train: locs_pred:{locs_pred[:5].detach()}, confs_pred:{labels_out[:5].detach()}, pred_scores:{pred_scores}")

        save_anchor_pic(image=imgs,
                        path="./output/train_pred.jpg",
                        bboxes=locs_pred.detach(),
                        bboxes_label=confs_pred.detach(),
                        bboxes_scores=pred_scores,
                        height=600,
                        width=600,
                        targets=bboxes,
                        targets_label=labels_out.detach(),
                        choice=choice)


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.scale_xy = 1.0 / dboxes.scale_xy  # 10
        self.scale_wh = 1.0 / dboxes.scale_wh  # 5

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # [num_anchors, 4] -> [4, num_anchors] -> [1, 4, num_anchors]
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)

        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        self.task2_loss = nn.CrossEntropyLoss()

    def _location_vec(self, loc):
        # type: (Tensor) -> Tensor
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc: anchor匹配到的对应GTBOX Nx4x8732
        :return:
        """

        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]  # Nx2x8732
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()  # Nx2x8732
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel, ptask2_label, gtask2_label):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
            ptask2_label, gtask2_label: Nx2, Nx2
        """
        # 获取正样本的mask  Tensor: [N, 8732]
        mask = torch.gt(glabel, 0)  # (gt: >)
        # mask1 = torch.nonzero(glabel)
        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = self.location_loss(ploc, vec_gd).sum(dim=1)  # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

        # hard negative mining Tenosr: [N, 8732]
        con = self.confidence_loss(plabel, glabel)

        # positive mask will never selected
        # 获取负样本
        con_neg = con.clone()
        con_neg[mask] = 0.0
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        # 第一次降序sort，置信度越高者，排在越前面,
        # 第二次对con_idx做升序排列，则排序结果又恢复为con_neg中的原有排序顺序，而每个位置对应在con_rank的值，
        # 记录的是其在con_idx的排序顺序，也就是元素序列值越小，confidence值越大，这样就可以直接mask con_neg，
        # 获取指定位置的元素, 比如：
        # >>> mask = torch.as_tensor([[True, False, False, True], [True, False, False, False]])
        # >>> mask
        # tensor([[True, False, False, True],
        #         [True, False, False, False]])
        # >>> con[mask]
        # tensor([0.2000, 0.3400, 0.5000])
        # >>> con_neg = con.clone()
        # >>> con_neg
        # tensor([[0.2000, 0.3000, 0.1200, 0.3400],
        #         [0.5000, 0.2000, 0.6000, 0.7000]])
        # >>> con_neg = con[mask]
        # >>> con_neg
        # tensor([0.2000, 0.3400, 0.5000])
        # >>> con_neg = con.clone()
        # >>> con_neg
        # tensor([[0.2000, 0.3000, 0.1200, 0.3400],
        #         [0.5000, 0.2000, 0.6000, 0.7000]])
        # >>> con_neg[mask] = 0
        # >>> con_neg
        # tensor([[0.0000, 0.3000, 0.1200, 0.0000],
        #         [0.0000, 0.2000, 0.6000, 0.7000]])
        # >>> _, con_idx = torch.sort(con_neg, descending=True)
        # >>> _
        # tensor([[0.3000, 0.1200, 0.0000, 0.0000],
        #         [0.7000, 0.6000, 0.2000, 0.0000]])
        # >>> con_idx
        # tensor([[1, 2, 0, 3],
        #         [3, 2, 1, 0]])
        # >>> _, con_rank = torch.sort(con_idx)
        # >>> _
        # tensor([[0, 1, 2, 3],
        #         [0, 1, 2, 3]])
        # >>> con_rank
        # tensor([[2, 0, 1, 3],
        #         [3, 2, 1, 0]])
        # >>> con * torch.lt(con_rank, 2)
        # tensor([[0.0000, 0.3000, 0.1200, 0.0000],
        #         [0.0000, 0.0000, 0.6000, 0.7000]])
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

        # number of negative three times positive
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * (mask.float() + neg_mask.float())).sum(dim=1)  # Tensor [N]

        # eg. [15, 3, 5, 0] -> [1.0, 1.0, 1.0, 0.0]
        num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况

        loc_loss = (loc_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        con_loss = (con_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失

        task2_loss = self.task2_loss(ptask2_label, gtask2_label)

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = 0.5 * (loc_loss + con_loss) + 0.5 * task2_loss

        #         print(f"---loss: con_loss:{con_loss:.4}, loc_loss:{loc_loss:.4}, task2_loss:{task2_loss:.4}, total_loss:{total_loss:.4}")

        return total_loss