UNet语义分割项目 - WHU-BuldingDataset数据集

项目简介

本项目针对WHU-BuldingDataset数据集进行了优化，这是一个基于遥感图像的建筑物分割数据集。

本项目是基于PyTorch实现的U-Net语义分割模型，用于图像分割任务。U-Net是一种经典的卷积神经网络架构，特别适用于医学图像分割和语义分割任务。

项目来源

原始代码基于开源项目 [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)（MIT协议）

本项目在原始代码基础上进行了深度的工程化改进，主要针对WHU-BuldingDataset数据集进行了优化，解决了训练效率、结果可复现性和预测可视化等问题。

✨ 核心改进 (Key Improvements)

针对原始代码，本项目完成了以下关键优化：

结果可复现 (Reproducibility)

- 在 PyTorch, NumPy 和 Python 中全面引入了固定随机种子 (seed=42)
- 确保每次划分训练集/验证集以及模型初始化的结果完全一致

验证逻辑修复 

- 逻辑修复：修复了进度条显示为批次而非样本数的Bug，并确保验证集所有样本都被评估（移除了 drop_last=True）

预测功能增强 (Enhanced Prediction)

- 输出四联可视化结果：原图 + 掩码(Mask) + 叠加图(Overlay) + 对比图(Result)
- 自动计算评价指标（面积占比、像素统计、均值标准差等）并保存为文本报告
- 结果自动按时间戳归档，便于管理和比较不同模型的预测结果
- 支持批量预测多张图片
- 提供可视化选项，实时查看预测结果

环境要求

- Python 3.6+
- PyTorch 1.0+
- 核心依赖：matplotlib, numpy, Pillow, tqdm, torchvision 等
- 详细列表见 requirements.txt

安装依赖

```bash
pip install -r requirements.txt
```

项目结构

```
Pytorch-UNet/
├── data/                 # 训练数据目录
│   ├── imgs/             # 训练图像
│   ├── masks/            # 图像掩码（标签）
│   └── test/             # 测试数据
├── unet/                 # UNet模型定义
│   ├── __init__.py
│   ├── unet_model.py     # 主模型结构
│   └── unet_parts.py     # 模型组件
├── utils/                # 工具函数
│   ├── dataset.py        # 数据集处理
│   └── data_vis.py       # 数据可视化
├── train.py              # 训练脚本 (已加入固定种子)
├── eval.py               # 评估脚本 (已修复计数逻辑)
├── predict.py            # 预测脚本 (已增强可视化与指标输出)
├── dice_loss.py          # 损失函数
├── requirements.txt      # 依赖列表
├── MODIFICATIONS.md      # 详细修改日志
├── prediction_results/   # 自动生成的预测结果目录
└── README.md             # 项目说明
```

使用说明

1. 准备数据

- 创建data/目录存放训练数据
- 数据集下载链接：[WHU-BuldingDataset.rar](https://pan.baidu.com/s/1Fy-qqjUjDjqPXUTu2A13uw?pwd=inv3)
- 数据集特点：遥感图像数据，包含清晰的建筑物轮廓，适合用于语义分割模型训练

2. 开始训练 (Training)

使用优化后的训练脚本启动训练（已内置固定种子 42）：

```bash
# 使用默认参数训练
python train.py

# 自定义训练参数
python train.py -e 30 -b 4 -l 0.0005 -s 1 -v 10
```

训练参数说明：
- `-e, --epochs`: 训练轮数（默认：5）
- `-b, --batch-size`: 批次大小（默认：1，根据GPU内存调整）
- `-l, --learning-rate`: 学习率（默认：0.1）
- `-s, --scale`: 图片缩放比例（默认：0.5）
- `-v, --validation`: 验证集比例（默认：10.0，即10%）
- `-f, --load`: 加载预训练模型（可选）

改进说明：
- 训练脚本每完成一个 Epoch 才会进行一次验证
- 数据集划分固定，重新运行训练会得到相同的划分结果
- 模型文件保存在 checkpoints/CP_epoch{epoch}.pth

3. 评估模型 (Evaluation)

```bash
# 使用最新模型评估
python eval.py --model checkpoints/CP_epoch10.pth --scale 0.5
```

4. 预测与可视化 (Prediction)

使用增强版预测脚本，生成可视化结果和指标报告：

```bash
# 下载预训练模型（可选）
# 预训练模型下载链接：[checkpoints.zip](https://pan.baidu.com/s/1a-RmIIveSshcmSnxJtkgXQ?pwd=u9s5)

# 单张图片预测
python predict.py -i data/test/2_998.JPG -m checkpoints/CP_epoch10.pth -s 1

# 多张图片预测
python predict.py -i data/test/2_998.JPG data/test/2_999.JPG -m checkpoints/CP_epoch10.pth -s 1 --viz

```

预测参数说明：
- `-i, --input`: 输入图片路径（必填，支持多张图片）
- `-m, --model`: 模型文件路径（默认：MODEL.pth）
- `-s, --scale`: 图片缩放比例（默认：0.5）
- `-t, --mask-threshold`: 掩码阈值（默认：0.5）
- `-v, --viz`: 可视化预测结果
- `-n, --no-save`: 不保存预测结果

输出结果结构：

脚本会在 `prediction_results/<时间戳>/` 目录下生成以下文件：

- `*_original_<timestamp>.jpg`: 原始输入图像
- `*_mask_<timestamp>.jpg`: 二值化掩码图
- `*_overlay_<timestamp>.jpg`: 掩码叠加在原图上的效果图（50%透明度，红色掩码）
- `*_result_<timestamp>.jpg`: 三联对比图（原图 + 掩码 + 叠加图）
- `*_metrics_<timestamp>.txt`: 预测指标报告，包含：
  - 掩码面积占比
  - 掩码像素数量
  - 掩码尺寸信息
  - 像素统计指标（均值、标准差等）
  - 模型和参数信息

5. 训练中断与恢复

- 训练过程中按 Ctrl+C 中断时，会自动保存当前模型为 `INTERRUPTED.pth`
- 可使用该文件恢复训练：
  ```bash
  python train.py -f INTERRUPTED.pth
  ```

6. 模型性能

使用WHU-BuldingDataset数据集进行训练后，模型在验证集上取得了良好的性能：

- **Validation Dice Coeff**: 0.8460421922827969
- 这个成绩表明模型能够准确地分割遥感图像中的建筑物，具有较高的分割精度
- 预训练模型下载链接：[checkpoints.zip](https://pan.baidu.com/s/1a-RmIIveSshcmSnxJtkgXQ?pwd=u9s5)

开源协议

本项目基于MIT协议开源，详见LICENSE文件。