# 深度学习实证研究 (DeepLearning-Empirical-Studies)

## 前言

本仓库创立于 2026 年 5 月 11 日，也算是我第一个决定长期更新和维护的仓库吧。在经历了持续约大半年的人工智能、计算机视觉、医学图像分割等领域的学习后，每当我学习一些新知识时都会产生一些奇奇怪怪的想法：“如果？那么？...”。因此本仓库主要用于记录作者学习机器学习/深度学习期间遇到的问题与想法。作者希望通过控制实验探究模型的表征现象，旨在从定性（未来也可能会定量）角度分析模型在不同实验条件下的表现差异，并思考其中的原因。期待从中得到启发，也希望作者的专业能力能与本仓库一样不断随着实践而进步。

## 组织架构

每个独立的实验主题均存放在单独的目录下。代码、实验报告与复现实验说明会保留在仓库中；本地数据集、模型权重、训练日志与中间输出文件默认不会上传至 GitHub。

```text
DeepLearning-Empirical-Studies/
├── Exp_01/                         # 实验一：数据规模、模型复杂度与输入信息量控制实验
│   ├── code/                       # 实验源代码
│   │   ├── train.py                # 训练入口
│   │   ├── model.py                # MLP、AlexNet、ResNet 系列模型定义
│   │   ├── load_data.py            # CIFAR-10/CIFAR-100 数据加载
│   │   ├── utils.py                # 日志、统计与权重保存工具
│   │   └── generate_figures/       # 论文图像生成脚本
│   ├── report/                     # 实验报告
│   │   ├── Report_cn.pdf           # 中文报告
│   │   └── Report_en.pdf           # 英文报告
│   ├── pre_exp.ipynb               # 预实验 notebook
│   ├── requirements.txt            # 当前实验依赖
│   └── README.md                   # 当前实验说明
├── Exp_02/                         # 后续实验主题
├── .gitignore                      # Git 忽略规则
├── LICENSE
└── README.md                       # 仓库主页
```

## 数据信息

- 本仓库当前实验主要使用 CIFAR-10 与 CIFAR-100 数据集。
- 数据集默认下载或放置在实验目录外的 `data/` 目录中，相关数据文件不会上传至 GitHub。
- 训练过程中产生的模型权重、SwanLab 日志、CSV 输出与论文中间图像默认属于本地产物，也不会随仓库提交。
- 若需要复现实验，请进入对应实验目录查看独立的 `README.md` 和 `requirements.txt`。

## 实验表格

下表收录了本仓库所有的实证探究项目。点击“实验名称”可进入对应目录查看源码、报告与复现实验说明。

| 实验编号 | 实验名称 | 探究内容 | 实验报告在线预览 | 完成状态 |
| :---: | :--- | :--- | :---: | :---: |
| `Exp-01` | [Data Scale, Model Complexity, and Input Information in Vision Generalization](./Exp_01/) | 通过 CIFAR-10/CIFAR-100 控制变量实验，观察训练数据规模、模型复杂度与输入信息模态对模型拟合能力和泛化性能的影响。 | 待补充（arXiv / 在线报告） | 进行中 |

## 联系方式

- 邮箱：zhoulyd@126.com

如果觉得有收获，不妨点击一个 ⭐ 哦~~~  
If you find this repository helpful, feel free to leave a ⭐.
