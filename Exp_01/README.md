# Exp-01：数据规模、模型复杂度与输入信息量控制实验

## 实验简介

本实验围绕深度视觉模型的泛化现象展开，主要观察三个因素对模型性能的影响：

1. 训练数据规模：比较不同训练样本数量下模型的训练损失、测试损失与测试准确率。
2. 模型复杂度：比较 MLP、AlexNet 与 ResNet 系列模型在相同数据条件下的表现。
3. 输入信息模态：比较 RGB、灰度化输入，以及额外拼接梯度、边缘和小波特征后的模型性能。

实验使用 CIFAR-10 与 CIFAR-100 数据集。所有实验尽量采用受控设置，不引入数据增强、权重衰减、Dropout 和学习率调度等额外正则化策略，以便更直接地观察数据规模、模型复杂度和输入信息本身带来的影响。

## 目录结构

```text
Exp_01/
├── code/
│   ├── train.py                    # 训练入口
│   ├── model.py                    # MLP、AlexNet、ResNet 模型定义
│   ├── load_data.py                # CIFAR-10/CIFAR-100 数据加载
│   ├── utils.py                    # SwanLab 日志、参数统计、权重保存等工具
│   └── generate_figures/           # 论文图像生成脚本
├── report/
│   ├── Report_cn.pdf               # 中文实验报告
│   └── Report_en.pdf               # 英文实验报告
├── pre_exp.ipynb                   # 一维非线性函数拟合预实验
├── requirements.txt                # Python 依赖
└── README.md
```

## 环境准备

建议使用 Conda 创建独立环境：

```bash
conda create -n exp01 python=3.10
conda activate exp01
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果需要使用 CUDA 版本的 PyTorch，建议根据本机 CUDA 版本参考 PyTorch 官网安装命令，然后再安装其余依赖。

## 数据集准备

代码支持自动下载 CIFAR-10 与 CIFAR-100 数据集。首次运行时可以加入 `--download` 参数：

```bash
python code/train.py --dataset cifar10 --model resnet18 --train-size 50000 --input-mode rgb --download
```

如果已经下载过数据集，后续运行可以去掉 `--download`。默认数据目录为 `./data`，该目录已在 `.gitignore` 中忽略，不会上传至 GitHub。

## 运行单组实验

以 CIFAR-10、ResNet-18、RGB 输入、50000 张训练样本为例：

```bash
python code/train.py --dataset cifar10 --model resnet18 --train-size 50000 --input-mode rgb
```

常用参数说明：

```text
--dataset       数据集，可选 cifar10 或 cifar100
--model         模型，可选 mlp、resnet18、resnet50 等
--train-size    训练样本数量
--input-mode    输入模式，可选 rgb、gray3、rgb_grad、rgb_edge、rgb_wavelet
--epochs        训练轮数
--batch-size    批大小
--lr            学习率
--swan-mode     SwanLab 记录模式，可选 cloud、offline、disabled
```

## 运行输入信息量控制实验

当前 `train.py` 中的 `--run-all` 会遍历预设模型和输入模态组合。示例：

```bash
python code/train.py --dataset cifar10 --train-size 50000 --run-all
```

若需要调整批量实验的模型或输入模式，可修改 `code/train.py` 中的：

```python
EXP1_MODELS = ["mlp", "resnet18", "resnet50"]
EXP2_INPUT_MODES = ["rgb", "gray3", "rgb_grad", "rgb_edge", "rgb_wavelet"]
```

## 输出结果

训练过程会记录以下指标：

- `train/loss`
- `train/acc`
- `test/loss`
- `test/acc`
- `lr`

实验结果、模型权重和图像输出默认保存在本地结果目录中。这些文件属于实验产物，默认不会上传至 GitHub。

## 论文图像生成

论文图像生成脚本位于：

```text
code/generate_figures/
```

运行示例：

```bash
python code/generate_figures/plot_resnet18_vs_resnet152_acc_curves.py
python code/generate_figures/plot_best_acc_by_data_scale.py
```

生成的图片默认保存到 `outputs/weights/figures/`。
