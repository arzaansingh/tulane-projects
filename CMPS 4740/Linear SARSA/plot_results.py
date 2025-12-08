import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

LOG_DIR = "logs"
PLOT_DIR = "plots"

def get_latest_log():
    """Finds the most recently modified CSV file in the logs directory."""
    list_of_files = glob.glob(os.path.join(LOG_DIR, '*.csv'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def plot_training():
    log_file = get_latest_log()
    if not log_file:
        print(f"No logs found in {LOG_DIR}. Run train_sarsa.py first.")
        return

    print(f"Plotting data from: {log_file}")
    
    try:
        data = pd.read_csv(log_file)
    except pd.errors.EmptyDataError:
        print("Log file is empty.")
        return

    # --- FILTER FOR LATEST RUN ---
    # If 'Episode' drops (e.g. 100 -> 1), it means a new training session started.
    # We slice the data to only show the latest session to avoid messy overlapping lines.
    if len(data) > 1 and 'Episode' in data.columns:
        resets = data.index[data['Episode'] < data['Episode'].shift(1)].tolist()
        if resets:
            last_reset_index = resets[-1]
            print(f"Detected previous runs in master log. Plotting only the latest run (Row {last_reset_index}+).")
            data = data.iloc[last_reset_index:]

    os.makedirs(PLOT_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # --- SUBPLOT 1: WIN RATE ---
    plt.subplot(1, 2, 1)
    if 'WinRate' in data.columns:
        plt.plot(data['Episode'], data['WinRate'], label='Overall Win Rate', color='blue')
    
    # Mark where the opponent switch happened (if visible in data)
    if 'Opponent' in data.columns:
        # Use a safe method to find changes in the Opponent column
        # compare current row to previous row
        opp_changes = data['Opponent'] != data['Opponent'].shift(1)
        # Filter strictly for changes that are NOT the first row of our slice
        # (The first row always looks like a "change" from NaN)
        switches = data.index[opp_changes].tolist()
        
        # We only care if a switch happens AFTER the start of our current plot data
        if len(switches) > 0:
            for switch_idx in switches:
                if switch_idx > data.index[0]:
                    switch_ep = data.loc[switch_idx, 'Episode']
                    plt.axvline(x=switch_ep, color='red', linestyle='--', label='Opponent Switch')
                    break # Just label the first switch we see to avoid clutter
    
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('Agent Win Rate (Current Session)')
    plt.legend()
    plt.grid(True)
    
    # --- SUBPLOT 2: EXPLORATION (Tau or Epsilon) ---
    plt.subplot(1, 2, 2)
    
    # Check which metric exists in the CSV
    if 'Tau' in data.columns:
        plt.plot(data['Episode'], data['Tau'], label='Temperature (Tau)', color='orange')
        plt.ylabel('Temperature (Softmax)')
        plt.title('Softmax Temperature Decay')
    elif 'Epsilon' in data.columns:
        plt.plot(data['Episode'], data['Epsilon'], label='Epsilon', color='orange')
        plt.ylabel('Epsilon (Greedy)')
        plt.title('Epsilon Decay')
    
    plt.xlabel('Episodes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot with same ID as log
    plot_name = log_file.replace("logs", "plots").replace(".csv", ".png")
    # Handle path differences if not replacing directly
    if "plots" not in plot_name:
        plot_name = os.path.join(PLOT_DIR, os.path.basename(log_file).replace(".csv", ".png"))

    plt.savefig(plot_name)
    print(f"Graph saved to {plot_name}")
    plt.show()

if __name__ == "__main__":
    plot_training()