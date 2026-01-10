import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# --- CONFIGURATION ---
LOG_DIR = "v16_logs"
PLOT_DIR = "v16_plots"
PLOT_FILE_PREFIX = "log"

def clean_percentage(val):
    if isinstance(val, str):
        return float(val.strip('%')) / 100.0
    return val

def get_log_file(opponent):
    return os.path.join(LOG_DIR, f"{PLOT_FILE_PREFIX}_{opponent}.csv")

def plot_training(opponent):
    log_file = get_log_file(opponent)
    if not os.path.exists(log_file):
        print(f"No log file found: {log_file}")
        return

    print(f"Plotting V16 data from: {log_file}")
    try: data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error: {e}")
        return

    if 'RollingWin' in data.columns and data['RollingWin'].dtype == object:
        data['RollingWin'] = data['RollingWin'].apply(clean_percentage)
    if 'OverallWin' in data.columns and data['OverallWin'].dtype == object:
        data['OverallWin'] = data['OverallWin'].apply(clean_percentage)

    x_col = 'Battles'
    if len(data) > 1 and x_col in data.columns:
        resets = data.index[data[x_col] < data[x_col].shift(1)].tolist()
        if resets:
            data = data.iloc[resets[-1]:]

    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(14, 10))
    
    # 1. Win Rates
    ax1 = plt.subplot(2, 2, 1)
    if 'RollingWin' in data.columns: ax1.plot(data[x_col], data['RollingWin'], color='blue', alpha=0.7, label='Rolling')
    if 'OverallWin' in data.columns: ax1.plot(data[x_col], data['OverallWin'], color='darkgreen', linewidth=2, label='Overall')
    ax1.set_title(f'V16 Win Rates vs {opponent.upper()}')
    ax1.legend()
    ax1.grid(True)

    # 2. Table Size
    ax2 = plt.subplot(2, 2, 2)
    if 'TableSize' in data.columns: ax2.plot(data[x_col], data['TableSize'], color='purple')
    ax2.set_title('Table Size')
    ax2.grid(True)

    # 3. Epsilon & Speed
    ax3 = plt.subplot(2, 2, 3)
    if 'Epsilon' in data.columns: 
        ax3.plot(data[x_col], data['Epsilon'], color='orange', label='Epsilon')
    ax3.set_title('Epsilon & Speed')
    ax3.legend(loc='upper left')
    
    ax3_twin = ax3.twinx()
    if 'Speed' in data.columns:
        ax3_twin.plot(data[x_col], data['Speed'], color='gray', linestyle='--', alpha=0.5, label='Speed')
    ax3_twin.legend(loc='upper right')
    
    # 4. Avg Reward
    ax4 = plt.subplot(2, 2, 4)
    if 'AvgReward' in data.columns: 
        ax4.plot(data[x_col], data['AvgReward'], color='red')
        ax4.axhline(0, color='black', linestyle='--')
    ax4.set_title('Average Reward')
    ax4.grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(PLOT_DIR, f"v16_results_{opponent}_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Saved: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("opponent", type=str, default="random")
    args = parser.parse_args()
    plot_training(args.opponent)