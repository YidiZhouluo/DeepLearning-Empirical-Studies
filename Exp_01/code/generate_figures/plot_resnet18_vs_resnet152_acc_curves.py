from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "ResNet18 vs ResNet152.csv"
OUTPUT_DIR = ROOT / "outputs" / "weights" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "fig_resnet18_vs_resnet152_acc_curves_cifar100.png"

MODELS = ["resnet18", "resnet152"]
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet152": "ResNet-152",
}

TRAIN_SIZE_ORDER = [5000, 10000, 20000, 30000, 50000]
SIZE_LABELS = {
    5000: "5k",
    10000: "10k",
    20000: "20k",
    30000: "30k",
    50000: "50k",
}

SIZE_COLORS = {
    5000: "#d62728",   # red
    10000: "#1f77b4",  # blue
    20000: "#f2c744",  # yellow
    30000: "#ff7f0e",  # orange
    50000: "#2ca02c",  # green
}

MODEL_LINESTYLES = {
    "resnet18": "-",
    "resnet152": "--",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def read_csv_columns() -> dict[str, list[float]]:
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for name in columns:
                value = row[name]
                columns[name].append(float(value) if value != "" else float("nan"))
    return columns


def metric_column(model: str, train_size: int) -> str:
    return f"exp1_cifar100_rgb_{model}_N{train_size}_seed42-test/acc_step"


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = read_csv_columns()
    epochs = data["step"]

    fig, ax = plt.subplots(figsize=(7.16, 3.05), constrained_layout=True)

    for train_size in TRAIN_SIZE_ORDER:
        for model in MODELS:
            column = metric_column(model, train_size)
            if column not in data:
                raise KeyError(f"Column not found: {column}")
            ax.plot(
                epochs,
                data[column],
                color=SIZE_COLORS[train_size],
                linestyle=MODEL_LINESTYLES[model],
                linewidth=1.15 if model == "resnet18" else 1.05,
                alpha=0.95 if model == "resnet18" else 0.82,
                label=f"{MODEL_LABELS[model]} {SIZE_LABELS[train_size]}",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Top-1 accuracy")
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(0.1, 0.7)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
