#!/usr/bin/env python3
"""Plot training curves from training_log.csv.

  py plot_training.py              one-shot window
  py plot_training.py --watch 5    live view, refresh every 5 s
  py plot_training.py --out fig.png
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SURFACE = "#16181d"
PANEL = "#1c1f26"
INK = "#d8dae0"
INK_MUTED = "#8a8d96"
GRID = "#31353f"
BLUE = "#6fb1ec"
AMBER = "#e3a94f"

GOAL_SERIES = (("score_uu", "#6fb1ec", "up-up"),
               ("score_ud", "#e3a94f", "up-down"),
               ("score_du", "#62d6c3", "down-up"),
               ("score_dd", "#b58ee6", "down-down"))


def style():
    sns.set_theme(style="darkgrid", rc={
        "figure.facecolor": SURFACE,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "text.color": INK,
        "axes.labelcolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.titlecolor": INK,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    })


def draw(fig, axes, df):
    window = max(1, len(df) // 25)

    ax = axes[0]
    ax.clear()
    series = df["avg_reward"].dropna()
    ax.plot(series.index, series, color=BLUE, alpha=0.3,
            linewidth=1.0, label="per iteration")
    smoothed = series.rolling(window, min_periods=1).mean()
    ax.plot(series.index, smoothed, color=BLUE, linewidth=2.0,
            label=f"rolling mean ({window})")
    ax.set_title("Average episode reward", loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.01)

    ax = axes[1]
    ax.clear()
    if "score_uu" in df.columns:
        for col, color, label in GOAL_SERIES:
            s = df[col].dropna()
            sm = s.rolling(window, min_periods=1).mean()
            ax.plot(s.index, s, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(s.index, sm, color=color, linewidth=2.0, label=label)
        ax.axhline(1.0, color=GRID, linewidth=1.0, linestyle="--")
        ax.set_ylim(-1.05, 1.1)
        ax.set_title("Goal satisfaction by commanded goal "
                     "(mean ½(g·cosθ), 1.0 = both poles on goal)",
                     loc="left", fontsize=11)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9, ncols=2)
    else:
        s = df["avg_len"].dropna()
        sm = s.rolling(window, min_periods=1).mean()
        ax.plot(s.index, s, color=AMBER, alpha=0.3, linewidth=1.0,
                label="per iteration")
        ax.plot(s.index, sm, color=AMBER, linewidth=2.0,
                label=f"rolling mean ({window})")
        ax.set_title("Average episode length (ticks)", loc="left", fontsize=11)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.margins(x=0.01)
    axes[1].set_xlabel("iteration")
    n = int(df["iter"].iloc[-1]) + 1 if len(df) else 0
    fig.suptitle(f"cart-pole training — {n} iterations", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="training_log.csv")
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="keep refreshing from the log")
    ap.add_argument("--out", help="write a PNG instead of opening a window")
    args = ap.parse_args()

    if args.out:
        matplotlib.use("Agg")
    style()

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    def load():
        if not os.path.exists(args.log) or os.path.getsize(args.log) < 40:
            return None
        return pd.read_csv(args.log)

    df = load()
    if df is None or df.empty:
        raise SystemExit(f"no data in {args.log} yet — start train.py first")
    draw(fig, axes, df)

    if args.out:
        fig.savefig(args.out, dpi=130, facecolor=SURFACE)
        print(f"wrote {args.out}")
        return

    if args.watch:
        plt.ion()
        plt.show()
        while plt.fignum_exists(fig.number):
            plt.pause(args.watch)
            df = load()
            if df is not None and not df.empty:
                draw(fig, axes, df)
                fig.canvas.draw_idle()
    else:
        plt.show()


if __name__ == "__main__":
    main()
