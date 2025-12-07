import subprocess
import sys
import time
import os
import csv

# --- CONFIG ---
TOTAL_BATTLES = 1000000 
BATCH_SIZE = 5000 

EPS_START = 0.5 
EPS_END = 0.05 
DECAY_BATTLES = 500000 

def get_last_stats(log_file):
    if not os.path.exists(log_file): return 0, 0
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) <= 1: return 0, 0
            last = data[-1]
            battles = int(last[0])
            win_pct = float(last[2].strip('%')) / 100.0
            return battles, int(battles * win_pct)
    except: return 0, 0

def get_epsilon(battles):
    if battles >= DECAY_BATTLES: return EPS_END
    return max(EPS_END, EPS_START - (battles / DECAY_BATTLES) * (EPS_START - EPS_END))

def main():
    opponent = "random" 
    if len(sys.argv) > 1: opponent = sys.argv[1]
    
    log_file = f"v10_logs/tabular_log_{opponent}.csv"
    
    print(f"🚀 STARTING V10 TABULAR vs {opponent.upper()}")
    
    while True:
        battles, wins = get_last_stats(log_file)
        if battles >= TOTAL_BATTLES: break
            
        eps = get_epsilon(battles)
        print(f"\n--- Launching (Bat {battles}, Eps {eps:.3f}) ---")
        
        cmd = [
            sys.executable, "train_tabular_v10.py",
            "--historic_battles", str(battles),
            "--historic_wins", str(wins),
            "--batch_size", str(BATCH_SIZE),
            "--epsilon", str(eps),
            "--opponent", opponent
        ]
        
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("⚠️ Crash. Restarting...")
            time.sleep(5)
        else:
            time.sleep(1)

if __name__ == "__main__":
    main()