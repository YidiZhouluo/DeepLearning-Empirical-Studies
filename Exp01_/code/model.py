"""
CIFAR-10/100 分类模型（32×32 输入）
- MLP：全连接基线
- AlexNet：卷积堆叠 + 全连接
- ResNet：适配 CIFAR 小图像的残差网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, hidden_dim=512, num_hidden_layers=3):
        super().__init__()
        input_dim = input_channels * 32 * 32

        layers = [nn.Flatten()]
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AlexNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNet(nn.Module):
    """
    适配 CIFAR-10/100 的 ResNet。
    相对 ImageNet 原版:
    1. 首层使用 3x3 卷积
    2. 不使用 maxpool
    3. 使用 adaptive average pooling

    block_type="basic" 对应 ResNet-18/34。
    block_type="bottleneck" 对应 ResNet-50/101/152。
    """

    def __init__(
        self,
        layers: list[int],
        block_type: str,
        num_classes: int = 10,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if block_type not in {"basic", "bottleneck"}:
            raise ValueError("block_type 必须是 'basic' 或 'bottleneck'")

        self.block_type = block_type
        self.entry_channels = 64
        self.expansion = 1 if block_type == "basic" else 4
        self.input_channels = self.entry_channels

        self.entry_block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                self.entry_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.entry_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(512, blocks=layers[3], stride=2)

        self.head = nn.Linear(512 * self.expansion, num_classes)

    def _basic_block(self, mid_channels: int, stride: int = 1) -> nn.Module:
        input_channels = self.input_channels
        output_channels = mid_channels * self.expansion

        conv_branch = nn.Sequential(
            nn.Conv2d(input_channels,mid_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels,output_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(output_channels),
        )

        if stride != 1 or input_channels != output_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            shortcut = nn.Identity()

        return _ResidualBlock(conv_branch, shortcut)

    def _bottleneck(self, mid_channels: int, stride: int = 1) -> nn.Module:
        input_channels = self.input_channels
        output_channels = mid_channels * self.expansion

        conv_branch = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        if stride != 1 or input_channels != output_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(input_channels,output_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(output_channels),
            )
        else:
            shortcut = nn.Identity()

        return _ResidualBlock(conv_branch, shortcut)

    def _make_layer(
        self,
        mid_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        block_builder = self._basic_block if self.block_type == "basic" else self._bottleneck

        layer_list = [block_builder(mid_channels, stride=stride)]
        self.input_channels = mid_channels * self.expansion

        for _ in range(1, blocks):
            layer_list.append(block_builder(mid_channels, stride=1))

        return nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


class _ResidualBlock(nn.Module):
    """残差单元：out = ReLU(F(x) + shortcut(x))"""

    def __init__(self, conv_branch: nn.Module, shortcut: nn.Module) -> None:
        super().__init__()
        self.conv_branch = conv_branch
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv_branch(x) + self.shortcut(x))


class ResNet18(ResNet):
    """
    适配 CIFAR-10/100 的 ResNet-18。
    结构：[2, 2, 2, 2] 个 BasicBlock；expansion=1。
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__(
            layers=[2, 2, 2, 2],
            block_type="basic",
            num_classes=num_classes,
            input_channels=input_channels,
        )


class ResNet34(ResNet):
    """
    适配 CIFAR-10/100 的 ResNet-34。
    结构：[3, 4, 6, 3] 个 BasicBlock；expansion=1。
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__(
            layers=[3, 4, 6, 3],
            block_type="basic",
            num_classes=num_classes,
            input_channels=input_channels,
        )


class ResNet50(ResNet):
    """
    适配 CIFAR-10/100 的 ResNet-50。
    结构：[3, 4, 6, 3] 个瓶颈块；expansion=4。
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__(
            layers=[3, 4, 6, 3],
            block_type="bottleneck",
            num_classes=num_classes,
            input_channels=input_channels,
        )


class ResNet101(ResNet):
    """
    适配 CIFAR-10/100 的 ResNet-101。
    结构：[3, 4, 23, 3] 个瓶颈块；expansion=4。
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__(
            layers=[3, 4, 23, 3],
            block_type="bottleneck",
            num_classes=num_classes,
            input_channels=input_channels,
        )


class ResNet152(ResNet):
    """
    适配 CIFAR-10/100 的 ResNet-152。
    结构：[3, 8, 36, 3] 个瓶颈块；expansion=4。
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__(
            layers=[3, 8, 36, 3],
            block_type="bottleneck",
            num_classes=num_classes,
            input_channels=input_channels,
        )

if __name__ == "__main__":
    x_rgb = torch.randn(2, 3, 32, 32)
    x_gray = torch.randn(2, 1, 32, 32)

    alex = AlexNet(num_classes=10, input_channels=3)
    res = ResNet152(num_classes=10, input_channels=3)
    res18 = ResNet18(num_classes=10, input_channels=3)
    print("AlexNet RGB :", alex(x_rgb).shape)
    print("ResNet152 RGB:", res(x_rgb).shape)
    print("ResNet18 RGB:", res18(x_rgb).shape)

    alex_g = AlexNet(num_classes=10, input_channels=1)
    res_g = ResNet152(num_classes=10, input_channels=1)
    print("AlexNet Gray :", alex_g(x_gray).shape)
    print("ResNet152 Gray:", res_g(x_gray).shape)
