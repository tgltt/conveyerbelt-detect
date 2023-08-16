# 传送带异物检测及物料跑偏检测挑战赛
<div>
    <b>一、赛事背景</b><br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在工业场景中，对传送带上的异物检测以及传送带物料偏移检测是保证产品质量以及生产安全的一个重要的环节。但通常面临着异物种类繁多、异物样本不规则、异物目标小、部分异物种类样本少的问题。本赛题就是针对此类场景设置，在限定样本数量的情况下进行传送带安全检测，即对传送带异物以及传送带物料是否存在偏移进行检测识别，是一个极具挑战性的小目标检测任务。<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;赛事网址：https://challenge.xfyun.cn/topic/info?type=conveyor-belt&option=ssgy
</div>
<br>
<div>
    <b>二、赛事任务</b> <br>
    本次赛事针对钢铁生产企业物料传送皮带图片，参赛者需要完成如下要求：   <br>
    1）将皮带上是否有异物检测出来，并标记异物类别和位置;    <br>
    2）检测皮带上物料是否跑偏，给出检测结果。<br>
    
    输入/输出: <br>
    (1) 输入:一张图片，jpg格式 <br>
    (2) 输出:检测产品的异物类型，异物位置，物料跑偏检测结果。 <br>
</div>
<br>
 
<li>传送带异物检测及物料跑偏检测数据集</b><br>
<div>
数据集地址：https://challenge.xfyun.cn/topic/info?type=conveyor-belt&option=stsj <br>
数据集页面：
<img width="1011" alt="image" src="https://github.com/tgltt/conveyerbelt-detect/assets/36066270/21c82328-ad15-4d52-80f4-ed22b58f7d6a">
</div>
<br>
<br>
    
<li><b>工程主要结构说明</b><br>

|    文件/文件夹         |                   功能描述               |
|       ----            |                    ----                 |
| kaggle_jupyter        | 存放本工程运行在jupyter notebook的实现文件 |
| paper                 | 存放模型相关的论文(ResNet、SSD)           |
| res50+ssd             | 存放ResNet50+SSD预训练模型                   |
| save_weights          | 存放模型每个迭代训练完成后的模型参数       |
| ssd300_model.py       | 存放SSD模改后的模型                      |
| train_eval_utils.py   | 每次迭代的训练和预测功能                 |
| train_ssd300.py       | 训练模型的主函数及入口类                  |
| my_dataset.py         | 定义批量访问数据集的功能类                 |
| transforms.py         | 预处理图片的功能类                        |
| predict_test.py       | 验证模型的推理功能类, 只验证图片           |
| calculate_mean_std.py | 计算训练集图像RGB三个通道的均值和方差, 由于数据集并非ImageNet、CoCo、Voc这类国际大型数据集，故数据集和标准差需自行计算 |
<br>
<br>

<li><b>环境配置</b><br>
<div>Pip 安装包含所有 requirements.txt 的包，环境要求 Python>=3.6，且 PyTorch>=1.7
此外，需安装Coco性能(mAP)评估工具，使用下面的命令:</div>
    
```bash
pip install pycocotools 
```
<br>
<br>  
    
<li><b>工程算法说明</b><br>
<div>
    本工程基于<b>多任务多尺度的物体检测框架SSD</b>，根据军用航空器数据集的特点，进行<b>算法模型改造</b>，涉及改造点为图片数据格式筛选、训练集/测试集数据划分、图像标签数据结构转换为COCO数据格式、学习率及学习率调度参数调整、根据军用飞行器数据集计算Normalization的均值及标准差等。<br>
</div>
    
<div align=center>
    <img width="737" alt="image" src="https://user-images.githubusercontent.com/36066270/219654243-874a721d-72c2-4a6b-a577-747701e9fa5d.png"><br>
    <img width="813" alt="image" src="https://user-images.githubusercontent.com/36066270/219656949-dd7f7981-2f2a-49ff-a6b2-553306c1c126.png"><br>
    <b>SSD原理图</b> 
</div>
<br>
<br>

<li><b>本工程运行效果</b><br>
    <img width="649" alt="image" src="https://github.com/tgltt/conveyerbelt-detect/assets/36066270/45a1fdcf-59e3-40fe-9e8a-748263c7fe49">

<li><b>模型性能(COCO评估)</b><br>
    数据集10, 000张图片, 405张用于训练，剩下182张图片用于测试和评估，采用GPU P100训练70小时, 总共经历119个迭代(epochs)，目前达到的性能Iou 0.50以上的mAP为55.0%。召回率为40.6%.<br><br>
    
<div align=center>
<b>模改后的算法模型性能指标(epochs=249)</b>

|          PR指标         |     IoU   |   area | maxDets| AP/AR取值 |
|          :-----         |   :----:  | :----: | :----: |  :----:  |
| Average Precision  (AP) | 0.50:0.95 |  all   |   100  |  0.321   |
| Average Precision  (AP) | 0.50      |  all   |   100  |  0.550   |
| Average Precision  (AP) | 0.75      |  all   |   100  |  0.325   |
| Average Precision  (AP) | 0.50:0.95 |  small |   100  | -1.000   |
| Average Precision  (AP) | 0.50:0.95 | medium |   100  |  0.289   |
| Average Precision  (AP) | 0.50:0.95 | large  |   100  |  0.350   |
| Average Recall     (AR) | 0.50:0.95 |  all   |   1    |  0.331   |
| Average Recall     (AR) | 0.50:0.95 |  all   |   10   |  0.372   |
| Average Recall     (AR) | 0.50:0.95 |  all   |   100  |  0.372   |
| Average Recall     (AR) | 0.50:0.95 |  small |   100  | -1.000   |
| Average Recall     (AR) | 0.50:0.95 | medium |   100  |  0.319   |
| Average Recall     (AR) | 0.50:0.95 | large  |   100  |  0.406   |

</div>
    
<br>

<br><br>
<li><b>展望</b><br>
本工程目前的检测精度还不够高，后续需对此进一步性能优化。
