import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys
import argparse
from datetime import datetime

# --- CONFIGURATION ---
LOG_DIR = "v4_logs"
PLOT_DIR = "v4_plots"

def get_log_file(opponent):
    """Determines the log file name based on the opponent argument."""
    return os.path.join(LOG_DIR, f"dqn_log_{opponent}.csv")

def plot_training(opponent):
    log_file = get_log_file(opponent)
    
    if not os.path.exists(log_file):
        print(f"No log file found for opponent '{opponent}' at {log_file}.")
        return

    print(f"Plotting DQN data from: {log_file}")
    
    try:
        data = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Filter for the latest continuous run session
    if len(data) > 1 and 'Episode' in data.columns:
        resets = data.index[data['Episode'] < data['Episode'].shift(1)].tolist()
        if resets:
            last_reset_index = resets[-1]
            print(f"Detected previous runs. Plotting latest session (Row {last_reset_index}+).")
            data = data.iloc[last_reset_index:]

    # Create figure with 3 rows, 2 columns (5 graphs + 1 empty space)
    plt.figure(figsize=(18, 16))
    
    # --- GRAPH 1: Rolling Win Rate (Alone) ---
    ax1 = plt.subplot(3, 2, 1)
    if 'RollingWin' in data.columns:
        ax1.plot(data['Episode'], data['RollingWin'], label='Rolling Win Rate', color='blue', linewidth=2)
    ax1.set_title(f'Rolling Win Rate (Last 1k Intervals)')
    ax1.set_ylabel('Win Rate')
    ax1.set_xlabel('Episodes')
    ax1.grid(True)
    ax1.legend()

    # --- GRAPH 2: Overall Win Rate (Alone) ---
    ax2 = plt.subplot(3, 2, 2)
    if 'OverallWin' in data.columns:
        ax2.plot(data['Episode'], data['OverallWin'], label='Overall Win Rate', color='darkgreen', linewidth=2)
    ax2.set_title(f'Overall Cumulative Win Rate vs {opponent.upper()}')
    ax2.set_ylabel('Win Rate')
    ax2.set_xlabel('Episodes')
    ax2.grid(True)
    ax2.legend()
    
    # --- GRAPH 3: Combined Win Rates ---
    ax3 = plt.subplot(3, 2, 3)
    if 'RollingWin' in data.columns:
        ax3.plot(data['Episode'], data['RollingWin'], label='Rolling Win', color='blue', alpha=0.7)
    if 'OverallWin' in data.columns:
        ax3.plot(data['Episode'], data['OverallWin'], label='Overall Win', color='darkgreen')
    ax3.set_title('Combined Win Rate Progression')
    ax3.set_ylabel('Win Rate')
    ax3.set_xlabel('Episodes')
    ax3.legend()
    ax3.grid(True)

    # --- GRAPH 4: Epsilon ---
    ax4 = plt.subplot(3, 2, 4)
    if 'Epsilon' in data.columns:
        ax4.plot(data['Episode'], data['Epsilon'], label='Epsilon', color='orange')
    ax4.set_title('Epsilon Decay (Exploration)')
    ax4.set_ylabel('Epsilon Value')
    ax4.set_xlabel('Episodes')
    ax4.legend()
    ax4.grid(True)

    # --- GRAPH 5: Speed ---
    ax5 = plt.subplot(3, 2, 5)
    if 'Speed' in data.columns:
        ax5.plot(data['Episode'], data['Speed'], label='Speed (bat/s)', color='purple')
    ax5.set_title('Training Speed')
    ax5.set_ylabel('Battles / Second')
    ax5.set_xlabel('Episodes')
    ax5.legend()
    ax5.grid(True)
    
    # Note: Subplot 6 is empty

    plt.tight_layout()
    
    # --- ADD TIMESTAMP TO FILENAME ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(PLOT_DIR, f"dqn_results_{opponent}_{timestamp}.png")
    
    plt.savefig(plot_path)
    print(f"Graph saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot DQN training results.")
    parser.add_argument("opponent", type=str, choices=["random", "maxbp", "heuristic"], help="Opponent used in the training log.")
    args = parser.parse_args()
    plot_training(args.opponent)