import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "tdmpc2" / "logs" / "mushr-nav-static" / "1" / "default"


def clean_numeric(series):
    """兼容 TD-MPC2 train.csv 中的 `tensor(-1.23)` 字符串。"""

    def parse(value):
        if isinstance(value, str):
            match = re.match(r"tensor\(([-+0-9.eE]+)\)", value)
            if match:
                return float(match.group(1))
        return value

    return pd.to_numeric(series.map(parse), errors="coerce")


def read_csv(path: Path):
    df = pd.read_csv(path)
    for col in df.columns:
        if col != "step":
            df[col] = clean_numeric(df[col])
    return df


def plot_curves(log_dir: Path, out_path: Path):
    eval_path = log_dir / "eval.csv"
    train_path = log_dir / "train.csv"
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    eval_df = read_csv(eval_path)
    train_df = read_csv(train_path)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160, sharex=False)
    ax = axes[0, 0]
    ax.plot(eval_df["step"], eval_df["episode_success"], marker="o", label="eval")
    ax.plot(train_df["step"], train_df["episode_success"].rolling(20, min_periods=1).mean(), label="train rolling20", alpha=0.8)
    ax.set_title("Success")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("success rate")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(eval_df["step"], eval_df["episode_reward"], marker="o", label="eval")
    ax.plot(train_df["step"], train_df["episode_reward"].rolling(20, min_periods=1).mean(), label="train rolling20", alpha=0.8)
    ax.set_title("Reward")
    ax.set_ylabel("episode reward")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(eval_df["step"], eval_df["episode_collision"], marker="o", label="eval")
    ax.plot(train_df["step"], train_df["episode_collision"].rolling(20, min_periods=1).mean(), label="train rolling20", alpha=0.8)
    ax.set_title("Collision")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("collision rate")
    ax.legend()

    ax = axes[1, 1]
    if len(eval_df) > 1:
        ax.plot(eval_df["step"], eval_df["episode_reward"].diff(), marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Eval Reward Delta")
    ax.set_xlabel("step")
    ax.set_ylabel("delta reward")

    for axis in axes.flat:
        axis.grid(True, linewidth=0.4, alpha=0.35)

    fig.suptitle(str(log_dir), fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return eval_df, train_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (args.log_dir / "training_curves.png")
    eval_df, train_df = plot_curves(args.log_dir, out)
    print(f"wrote plot to: {out}")
    print("eval summary:")
    print(eval_df.tail().to_string(index=False))
    print("train rolling20 tail:")
    cols = ["step", "episode_reward", "episode_success", "episode_collision"]
    tail = train_df[cols].copy()
    for col in cols[1:]:
        tail[col] = tail[col].rolling(20, min_periods=1).mean()
    print(tail.tail().to_string(index=False))


if __name__ == "__main__":
    main()
