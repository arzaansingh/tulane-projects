import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

LOG_DIR = "v10_logs"
PLOT_DIR = "v10_plots"

def clean_pct(val):
    if isinstance(val, str): return float(val.strip('%')) / 100.0
    return val

def plot_training(opponent):
    log_file = os.path.join(LOG_DIR, f"tabular_log_{opponent}.csv")
    if not os.path.exists(log_file): return

    try: data = pd.read_csv(log_file)
    except: return

    if 'RollingWin' in data.columns and data['RollingWin'].dtype == object:
        data['RollingWin'] = data['RollingWin'].apply(clean_pct)
    if 'OverallWin' in data.columns and data['OverallWin'].dtype == object:
        data['OverallWin'] = data['OverallWin'].apply(clean_pct)

    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(18, 12))
    
    # 1. Win Rate
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(data['Battles'], data['RollingWin'], color='blue', label='Rolling')
    ax1.plot(data['Battles'], data['OverallWin'], color='green', label='Overall')
    ax1.set_title(f'V10 Tabular Win Rates vs {opponent.upper()}')
    ax1.legend()
    ax1.grid(True)

    # 2. Table Size (Exploration of State Space)
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(data['Battles'], data['TableSize'], color='purple', label='States Explored')
    ax2.set_title('Q-Table Growth')
    ax2.set_ylabel('Unique States')
    ax2.grid(True)
    
    # 3. Epsilon
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(data['Battles'], data['Epsilon'], color='orange')
    ax3.set_title('Epsilon')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"v10_results_{opponent}.png"))
    print("Plot Saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("opponent", type=str, default="random")
    args = parser.parse_args()
    plot_training(args.opponent)