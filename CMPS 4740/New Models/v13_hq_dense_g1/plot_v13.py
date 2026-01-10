import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

# --- CONFIGURATION ---
LOG_DIR = "v13_logs"
PLOT_DIR = "v13_plots"
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

    print(f"Plotting V13 data from: {log_file}")
    
    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    # --- DATA CLEANING ---
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

    # --- SETUP 2x2 GRID ---
    plt.figure(figsize=(14, 10))
    
    # --- 1. Top Left: WIN RATES ---
    ax1 = plt.subplot(2, 2, 1)
    if 'RollingWin' in data.columns:
        ax1.plot(data[x_col], data['RollingWin'], label='Rolling Win Rate', color='blue', alpha=0.7)
    if 'OverallWin' in data.columns:
        ax1.plot(data[x_col], data['OverallWin'], label='Overall Win Rate', color='darkgreen', linewidth=2)
    ax1.set_title(f'V13 Win Rates vs {opponent.upper()}')
    ax1.set_ylabel('Win Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # --- 2. Top Right: Q-TABLE GROWTH ---
    ax2 = plt.subplot(2, 2, 2)
    if 'TableSize' in data.columns:
        ax2.plot(data[x_col], data['TableSize'], label='Unique States', color='purple')
    ax2.set_title('Q-Table Complexity (States Explored)')
    ax2.set_ylabel('Table Size (Count)')
    ax2.grid(True)

    # --- 3. Bottom Left: EPSILON & SPEED ---
    ax3 = plt.subplot(2, 2, 3)
    if 'Epsilon' in data.columns:
        ax3.plot(data[x_col], data['Epsilon'], label='Epsilon', color='orange')
    ax3.set_title('Exploration vs Speed')
    ax3.set_ylabel('Epsilon Value')
    ax3.set_xlabel('Battles')
    ax3.legend(loc='upper left')
    
    # Create a twin axis for speed
    ax3_twin = ax3.twinx()
    if 'Speed' in data.columns:
        ax3_twin.plot(data[x_col], data['Speed'], label='Speed (bat/s)', color='gray', linestyle='--', alpha=0.5)
    ax3_twin.set_ylabel('Speed (bat/s)')
    ax3_twin.legend(loc='upper right')

    # --- 4. Bottom Right: AVERAGE REWARD ---
    ax4 = plt.subplot(2, 2, 4)
    if 'AvgReward' in data.columns:
        ax4.plot(data[x_col], data['AvgReward'], label='Avg Reward (Per Battle)', color='red', alpha=0.8)
        
        # Add a zero line for reference
        ax4.axhline(0, color='black', linewidth=1, linestyle='--')
        
    ax4.set_title('Average Reward Trend')
    ax4.set_ylabel('Reward Value')
    ax4.set_xlabel('Battles')
    ax4.legend(loc='upper left')
    ax4.grid(True)

    plt.tight_layout()
    
    # --- SAVE ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(PLOT_DIR, f"v13_results_{opponent}_{timestamp}.png")
    
    plt.savefig(plot_path)
    print(f"Graph saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot V13 Tabular results.")
    parser.add_argument("opponent", type=str, default="random", help="Opponent used in the training log (e.g., random, maxbp).")
    args = parser.parse_args()
    plot_training(args.opponent)