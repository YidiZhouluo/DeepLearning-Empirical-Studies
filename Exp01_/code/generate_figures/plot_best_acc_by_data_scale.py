from pathlib import Path
import csv

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "outputs" / "test_loss_mlp_resnet18.csv"
OUTPUT_DIR = ROOT / "outputs" / "weights" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "fig_best_acc_by_data_scale.png"

MODELS = ["mlp", "resnet18"]
MODEL_LABELS = {
    "mlp": "MLP",
    "resnet18": "ResNet-18",
}

TRAIN_SIZE_ORDER = [5000, 20000, 50000]
SIZE_LABELS = {
    5000: "5k",
    20000: "20k",
    50000: "50k",
}

SIZE_COLORS = {
    5000: "#d62728",   # red
    20000: "#f2c744",  # yellow
    50000: "#2ca02c",  # green
}

MODEL_LINESTYLES = {
    "mlp": "--",
    "resnet18": "-",
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
    return f"exp1_cifar10_rgb_{model}_N{train_size}_seed42-test/loss_step"


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
                alpha=0.95 if model == "resnet18" else 0.86,
                label=f"{MODEL_LABELS[model]} {SIZE_LABELS[train_size]}",
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test loss")
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
