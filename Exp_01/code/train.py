import argparse
import copy
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from load_data import CIFAR10, CIFAR100
from model import MLP, AlexNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils import SwanLabLogger, StatisticsLogger, save_model_weights


EXP1_MODELS = ["mlp", "resnet18", "resnet50"]
EXP2_INPUT_MODES = ["rgb", "gray3", "rgb_grad", "rgb_edge", "rgb_wavelet"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, criterion, device, optimizer=None, desc="Train"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        bar = tqdm(loader, desc=desc, leave=False)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = y.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item()
            total += n
            bar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def build_model(model_name: str, num_classes: int = 10, input_channels: int = 3):
    model_name = model_name.lower()

    if model_name == "mlp":
        return MLP(input_channels=input_channels, num_classes=num_classes)
    if model_name == "alexnet":
        return AlexNet(input_channels=input_channels, num_classes=num_classes)
    if model_name == "resnet18":
        return ResNet18(input_channels=input_channels, num_classes=num_classes)
    if model_name == "resnet34":
        return ResNet34(input_channels=input_channels, num_classes=num_classes)
    if model_name == "resnet50":
        return ResNet50(input_channels=input_channels, num_classes=num_classes)
    if model_name == "resnet101":
        return ResNet101(input_channels=input_channels, num_classes=num_classes)
    if model_name == "resnet152":
        return ResNet152(input_channels=input_channels, num_classes=num_classes)

    raise ValueError(f"Unknown model: {model_name}")


INPUT_MODE_CHANNELS = {
    "rgb": 3,
    "gray3": 3,
    "rgb_grad": 4,
    "rgb_edge": 4,
    "rgb_wavelet": 15,
}


class PriorFeatureTransform:
    def __init__(self, input_mode: str):
        if input_mode not in INPUT_MODE_CHANNELS:
            raise ValueError(f"Unknown input_mode: {input_mode}")
        self.input_mode = input_mode
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        rgb = self.to_tensor(image)

        if self.input_mode == "rgb":
            return self._normalize(rgb)

        gray = self._rgb_to_gray(rgb)
        if self.input_mode == "gray3":
            return self._normalize(gray.repeat(3, 1, 1))

        if self.input_mode == "rgb_grad":
            grad = self._sobel_magnitude(gray)
            return self._normalize(torch.cat([rgb, grad], dim=0))

        if self.input_mode == "rgb_edge":
            grad = self._sobel_magnitude(gray)
            edge = (grad > (grad.mean() + 0.5 * grad.std())).float()
            return self._normalize(torch.cat([rgb, edge], dim=0))

        if self.input_mode == "rgb_wavelet":
            wavelet = self._haar_wavelet_features(rgb)
            return self._normalize(torch.cat([rgb, wavelet], dim=0))

        raise ValueError(f"Unknown input_mode: {self.input_mode}")

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    @staticmethod
    def _rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
        weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
        return (rgb * weights).sum(dim=0, keepdim=True)

    @staticmethod
    def _minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return (x - x.amin(dim=(-2, -1), keepdim=True)) / (
            x.amax(dim=(-2, -1), keepdim=True) - x.amin(dim=(-2, -1), keepdim=True) + eps
        )

    def _sobel_magnitude(self, gray: torch.Tensor) -> torch.Tensor:
        kernel_x = gray.new_tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]).view(1, 1, 3, 3)
        kernel_y = gray.new_tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ]).view(1, 1, 3, 3)

        x = gray.unsqueeze(0)
        grad_x = torch.nn.functional.conv2d(x, kernel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(x, kernel_y, padding=1)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12).squeeze(0)
        return self._minmax(magnitude)

    def _haar_wavelet_features(self, rgb: torch.Tensor) -> torch.Tensor:
        x00 = rgb[:, 0::2, 0::2]
        x01 = rgb[:, 0::2, 1::2]
        x10 = rgb[:, 1::2, 0::2]
        x11 = rgb[:, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.25
        lh = (x00 - x01 + x10 - x11) * 0.25
        hl = (x00 + x01 - x10 - x11) * 0.25
        hh = (x00 - x01 - x10 + x11) * 0.25
        features = torch.cat([ll, lh.abs(), hl.abs(), hh.abs()], dim=0).unsqueeze(0)
        features = torch.nn.functional.interpolate(
            features,
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return self._minmax(features)


def build_transform(input_mode: str):
    return PriorFeatureTransform(input_mode)


def build_dataset(dataset_name: str, data_path: str, transform, download: bool):
    if dataset_name == "cifar10":
        return CIFAR10(data_path, transform=transform, download_from_web=download), 10
    if dataset_name == "cifar100":
        return CIFAR100(data_path, transform=transform, download_from_web=download), 100

    raise ValueError(f"Unknown dataset: {dataset_name}")


def sample_image_grid(loader, max_n=4):
    images, _ = next(iter(loader))
    if images.size(1) > 3:
        images = images[:, :3]
    return SwanLabLogger.make_cls_grid(images, max_n=max_n)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model", choices=EXP1_MODELS, default="resnet18")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--input-mode",choices=list(INPUT_MODE_CHANNELS.keys()),default="rgb",)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data", default="./data")
    parser.add_argument("--download", action="store_true", help="是否从网络下载数据集")
    parser.add_argument("--swan-mode", choices=["cloud", "offline", "disabled"], default="cloud")
    parser.add_argument("--project", default="Github_Exp01")
    parser.add_argument("--log-images-every", type=int, default=10, help="每隔多少个 epoch 记录一次训练图像样本到日志中，0 表示不记录")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--weights-dir", default="./weights")
    parser.add_argument("--run-all", action="store_true", help="运行实验一的全部模型和训练集大小组合",default=True)
    return parser.parse_args()


def run_one_experiment(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = INPUT_MODE_CHANNELS[args.input_mode]
    experiment_name = (
        f"exp1_{args.dataset}_{args.input_mode}_"
        f"{args.model}_N{args.train_size}_seed{args.seed}"
    )

    transform = build_transform(args.input_mode)
    data, num_classes = build_dataset(args.dataset, args.data, transform, args.download)
    data.process_dataset()
    train_loader, test_loader = data.load_subset_dataset(
        num_samples=args.train_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        input_channels=input_channels,
    )

    stats_logger = StatisticsLogger()
    model_stats = stats_logger.get_statistics(
        model=model,
        model_name=args.model,
        input_channels=input_channels,
        image_size=32,
        device=device,
    )

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args) | {
        "device": str(device),
        "num_gpus": torch.cuda.device_count(),
        "num_classes": num_classes,
        "input_channels": input_channels,
        "params": model_stats["params"],
        "params_million": model_stats["params_million"],
        "flops": model_stats["flops"],
        "flops_gflops": model_stats["flops_gflops"],
    }

    logger = SwanLabLogger()
    logger.init(project_name=args.project, mode=args.swan_mode)
    logger.create_experiment(
        experiment_name=experiment_name,
        config=config,
        tags=[
            "exp1",
            args.dataset,
            args.input_mode,
            args.model,
            f"N={args.train_size}",
            f"seed={args.seed}",
        ],
    )

    try:
        best_test_acc = 0.0
        best_epoch = 0
        final_train_loss = final_train_acc = 0.0
        final_test_loss = final_test_acc = 0.0

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                f"Train {epoch}/{args.epochs}",
            )
            test_loss, test_acc = run_epoch(
                model,
                test_loader,
                criterion,
                device,
                None,
                f"Test  {epoch}/{args.epochs}",
            )

            final_train_loss = train_loss
            final_train_acc = train_acc
            final_test_loss = test_loss
            final_test_acc = test_acc

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch

            logger.log_metrics({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if args.log_images_every > 0 and epoch % args.log_images_every == 0:
                logger.log_image(
                    key="train/images",
                    image=sample_image_grid(train_loader, max_n=8),
                    step=epoch,
                    caption=f"Epoch {epoch}",
                )

        result_row = {
            "experiment_name": experiment_name,
            "dataset": args.dataset,
            "input_mode": args.input_mode,
            "model": args.model,
            "train_size": args.train_size,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_classes": num_classes,
            "num_gpus": torch.cuda.device_count(),
            "params_million": model_stats["params_million"],
            "flops_gflops": model_stats["flops_gflops"],
            "final_train_loss": final_train_loss,
            "final_train_acc": final_train_acc,
            "final_test_loss": final_test_loss,
            "final_test_acc": final_test_acc,
            "best_test_acc": best_test_acc,
            "best_epoch": best_epoch,
        }

        weights_path = Path(args.weights_dir) / args.dataset / args.input_mode / f"{experiment_name}.pt"
        save_model_weights(
            model,
            str(weights_path),
            metadata=result_row | config,
        )

        csv_path = Path(args.results_dir) / "exp1_results.csv"
        logger.save_csv_row(str(csv_path), result_row)
    finally:
        logger.finish()


def main():
    args = parse_args()

    if not args.run_all:
        run_one_experiment(args)
        return

    total = len(EXP1_MODELS) * len(EXP2_INPUT_MODES)
    current = 0
    for model_name in EXP1_MODELS:
        for input_mode in EXP2_INPUT_MODES:
            current += 1
            run_args = copy.deepcopy(args)
            run_args.model = model_name
            run_args.input_mode = input_mode
            print(
                f"\n[Experiment Matrix] Running {current}/{total}: "
                f"model={model_name}, input_mode={input_mode}, "
                f"dataset={run_args.dataset}, train_size={run_args.train_size}, seed={run_args.seed}\n"
            )
            run_one_experiment(run_args)


if __name__ == "__main__":
    main()
