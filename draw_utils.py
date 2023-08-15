import os
import torch

from torchvision.transforms import ToPILImage

from PIL import Image
from PIL.Image import fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor

import datetime
import matplotlib.pyplot as plt

import numpy as np

from utils import make_dir
from utils import device

STANDARD_COLORS = [
    'White', 'Cyan', 'Yellow', 'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Wheat', 'WhiteSmoke', 'YellowGreen',
    'Beige', 'Bisque', 'Violet', 'Green', 'Red'
]


def draw_text(draw,
              box: list,
              cls: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    将目标边界框和类别信息绘制到图片上
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()

    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}: {int(100 * score)}%"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))


def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = False):
    """
    将目标边界框信息，类别信息，mask信息绘制在图片上
    Args:
        image: 需要绘制的图片
        boxes: 目标边界框信息
        classes: 目标类别信息
        scores: 目标概率信息
        masks: 目标mask信息
        category_index: 类别与名称字典
        box_thresh: 过滤的概率阈值
        mask_thresh:
        line_thickness: 边界框宽度
        font: 字体类型
        font_size: 字体大小
        draw_[boxes]_on_image:
        draw_masks_on_image:

    Returns:

    """

    # 过滤掉低概率的目标
    idxs = np.greater(scores, box_thresh)

    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]

    if draw_boxes_on_image:
        draw_boxes = []
        for box, cls, score, color in zip(boxes, classes, scores, colors):
            draw_boxes.append((box, cls, score, color))

        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        for draw_box in draw_boxes[-1::-1]:
            left, top, right, bottom = draw_box[0]
            # 绘制目标边界框
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=draw_box[3])

            if category_index is not None:
                # 绘制类别和概率信息
                draw_text(draw, draw_box[0].tolist(), int(draw_box[1]), float(draw_box[2]), category_index, draw_box[3],
                          font, font_size)

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return image


def draw_image(img_file, predict_boxes, predict_classes, predict_scores, category_index,
               show_image_flag=True, save_image_flag=False, save_image_dir=None):
    original_img = Image.open(img_file)

    plot_img = draw_objs(original_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         category_index=category_index,
                         box_thresh=0.5,
                         line_thickness=2,
                         font='simsun.ttc',
                         font_size=20)
    if show_image_flag:
        plt.imshow(plot_img)
        plt.show()

    if save_image_flag:
        make_dir(path=save_image_dir)
        save_file = os.path.join(save_image_dir, os.path.basename(img_file))

        plot_img.save(save_file)


def plot_loss_and_lr(train_loss, learning_rate):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./loss_and_lr{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('epoch')
        plt.yticks([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
        plt.ylabel('mAP')
        plt.title('Eval mAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./mAP.png')
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)


def save_pic(image, path, bboxes=None, labels=None, height=None, width=None, category_index=None, bboxes_scores=None):
    make_dir(path)
    image.save(path)

    if bboxes is not None and labels is not None:
        bboxes_out = bboxes.copy()

        bboxes_out[:, 0], bboxes_out[:, 2] = bboxes[:, 0] * width, bboxes[:, 2] * width
        bboxes_out[:, 1], bboxes_out[:, 3] = bboxes[:, 1] * height, bboxes[:, 3] * height

        gt_scores = np.array([1.0 for i in range(len(labels))]) if bboxes_scores is None else bboxes_scores

        draw_image(path, bboxes_out, labels, gt_scores, category_index, save_image_flag=True,
                   save_image_dir=os.path.dirname(path))


def save_anchor_pic(image, path, bboxes, height, width, targets=None, targets_label=None, choice=None,
                    bboxes_label=None, bboxes_scores=None):
    toPilImage = ToPILImage()
    if choice is None:
        choice = 0
    image = toPilImage(image[choice, :, :, :])

    labels = np.array([0 for _ in range(len(bboxes))]) if bboxes_label is None else bboxes_label

    if targets is not None:
        bboxes = torch.cat((targets, bboxes.to(device=device)), dim=0).to(device=device)
        color_count = len(STANDARD_COLORS)
        targets_label = [color_count - 1 for _ in
                         range(len(targets))] if targets_label is None else color_count - targets_label

        labels = np.append(targets_label.cpu(), labels.cpu())

    if bboxes_scores is not None:
        scores = []

        bboxes_scores = bboxes_scores.tolist()
        target_scores = [1 for _ in range(len(targets))]

        scores.extend(target_scores)
        scores.extend(bboxes_scores)

        bboxes_scores = np.array(scores)

    save_pic(image=image, path=path, bboxes=bboxes.cpu().numpy(), labels=labels, height=height, width=width,
             bboxes_scores=bboxes_scores)