import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# --- CONFIGURATION ---
LOG_DIR = "v11_logs"
PLOT_DIR = "v11_plots"
PLOT_FILE_PREFIX = "tabular_log"

def clean_percentage(val):
    """Converts string '45.2%' to float 0.452"""
    if isinstance(val, str):
        return float(val.strip('%')) / 100.0
    return val

def get_log_file(opponent):
    """Determines the log file name based on the opponent argument."""
    return os.path.join(LOG_DIR, f"{PLOT_FILE_PREFIX}_{opponent}.csv")

def plot_training(opponent):
    log_file = get_log_file(opponent)
    
    if not os.path.exists(log_file):
        print(f"No log file found for opponent '{opponent}' at {log_file}.")
        return

    print(f"Plotting V11 Tabular data from: {log_file}")
    
    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    # --- DATA CLEANING ---
    # Convert win rates from percentage string to float
    if 'RollingWin' in data.columns and data['RollingWin'].dtype == object:
        data['RollingWin'] = data['RollingWin'].apply(clean_percentage)
    
    if 'OverallWin' in data.columns and data['OverallWin'].dtype == object:
        data['OverallWin'] = data['OverallWin'].apply(clean_percentage)

    x_col = 'Battles'

    # Filter for restarts (if Battles count drops, it means a restart happened)
    if len(data) > 1 and x_col in data.columns:
        resets = data.index[data[x_col] < data[x_col].shift(1)].tolist()
        if resets:
            last_reset_index = resets[-1]
            print(f"Detected previous runs. Plotting latest session (Row {last_reset_index}+).")
            data = data.iloc[last_reset_index:]

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Create figure with 3 rows, 1 column
    plt.figure(figsize=(12, 12))
    
    # --- GRAPH 1: Win Rates ---
    ax1 = plt.subplot(3, 1, 1)
    if 'RollingWin' in data.columns:
        ax1.plot(data[x_col], data['RollingWin'], label='Rolling Win Rate', color='blue', alpha=0.7)
    if 'OverallWin' in data.columns:
        ax1.plot(data[x_col], data['OverallWin'], label='Overall Win Rate', color='darkgreen', linewidth=2)
    ax1.set_title(f'V11 Win Rates vs {opponent.upper()}')
    ax1.set_ylabel('Win Rate')
    ax1.legend()
    ax1.grid(True)

    # --- GRAPH 2: Q-Table Growth (Complexity) ---
    ax2 = plt.subplot(3, 1, 2)
    if 'TableSize' in data.columns:
        ax2.plot(data[x_col], data['TableSize'], label='Unique State-Action Pairs', color='purple')
    ax2.set_title('Q-Table Growth (States Explored)')
    ax2.set_ylabel('Table Size (Count)')
    ax2.grid(True)
    
    # --- GRAPH 3: Epsilon & Speed ---
    ax3 = plt.subplot(3, 1, 3)
    if 'Epsilon' in data.columns:
        ax3.plot(data[x_col], data['Epsilon'], label='Epsilon (Exploration)', color='orange')
    ax3.set_title('Epsilon Decay and Training Speed')
    ax3.set_ylabel('Epsilon Value')
    ax3.set_xlabel(x_col)
    ax3.legend(loc='upper left')
    
    ax4 = ax3.twinx()
    if 'Speed' in data.columns:
        ax4.plot(data[x_col], data['Speed'], label='Speed (bat/s)', color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Speed (bat/s)')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    
    # --- ADD TIMESTAMP TO FILENAME (MANDATORY) ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(PLOT_DIR, f"v11_results_{opponent}_{timestamp}.png")
    
    plt.savefig(plot_path)
    print(f"Graph saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot V11 Tabular Q-Learning results.")
    # Note: Using heuristic as a placeholder, user can pass maxbp or random
    parser.add_argument("opponent", type=str, default="heuristic", help="Opponent used in the training log (e.g., random, maxbp).")
    args = parser.parse_args()
    plot_training(args.opponent)