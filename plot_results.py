import pandas as pd
import matplotlib.pyplot as plt
import os
import time

BASE = os.path.join(os.path.dirname(__file__), 'tdmpc2/logs')
STATIC_CSV = os.path.join(BASE, 'walker-walk-static-obstacle/1/default/train.csv')
DYNAMIC_CSV = os.path.join(BASE, 'walker-walk-dynamic-obstacle/1/default/train.csv')
REFRESH = 15

def load(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['episode_reward'] = pd.to_numeric(
        df['episode_reward'].astype(str).str.extract(r'([-\d.]+)')[0], errors='coerce')
    return df

plt.ion()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax0, ax1 = axes[0]
ax2, ax3 = axes[1]

while True:
    for ax in [ax0, ax1, ax2, ax3]:
        ax.cla()

    datasets = [
        (STATIC_CSV,  'Static',  'steelblue'),
        (DYNAMIC_CSV, 'Dynamic', 'tomato'),
    ]
    window = 20

    for path, label, color in datasets:
        df = load(path)
        if df is None:
            continue

        # Reward 滑动平均
        smoothed = df['episode_reward'].rolling(window, min_periods=1).mean()
        ax0.plot(df['step'], smoothed, label=label, color=color)
        ax0.fill_between(df['step'],
                         df['episode_reward'].rolling(window, min_periods=1).min(),
                         df['episode_reward'].rolling(window, min_periods=1).max(),
                         alpha=0.15, color=color)

        # Success 累加
        ax1.plot(df['step'], df['episode_success'].cumsum(), label=label, color=color)

        # Collision 滑动平均
        col_smooth = df['episode_collision'].rolling(window, min_periods=1).mean()
        ax2.plot(df['step'], col_smooth, label=label, color=color)
        ax2.fill_between(df['step'],
                         df['episode_collision'].rolling(window, min_periods=1).min(),
                         df['episode_collision'].rolling(window, min_periods=1).max(),
                         alpha=0.15, color=color)

        # Collision 累加
        ax3.plot(df['step'], df['episode_collision'].cumsum(), label=label, color=color)

    ax0.set_title(f'Reward (rolling avg window={window})')
    ax0.set_xlabel('Steps'); ax0.set_ylabel('Episode Reward')
    ax0.legend(); ax0.grid(alpha=0.3)

    ax1.set_title('Cumulative Success Count')
    ax1.set_xlabel('Steps'); ax1.set_ylabel('Cumulative Successes')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_title(f'Collision per Episode (rolling avg window={window})')
    ax2.set_xlabel('Steps'); ax2.set_ylabel('Collisions')
    ax2.legend(); ax2.grid(alpha=0.3)

    ax3.set_title('Cumulative Collision Count')
    ax3.set_xlabel('Steps'); ax3.set_ylabel('Cumulative Collisions')
    ax3.legend(); ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'results.png'), dpi=150)
    fig.canvas.draw()
    plt.pause(0.1)
    time.sleep(REFRESH)
