import subprocess
import sys
import time
import os
import csv

# --- CONFIG ---
TOTAL_BATTLES = 10000000 
BATCH_SIZE = 10000 

EPS_START = 0.5
EPS_END = 0.01
DECAY_BATTLES = 3500000 #2500000 

def get_last_stats(log_file):
    if not os.path.exists(log_file): return 0, 0
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) <= 1: return 0, 0
            
            last = data[-1]
            # CSV: [Battles, RollingWin, OverallWin, Epsilon, Speed, AvgReward, TableSize, Opponent]
            battles = int(last[0])
            pct_str = last[2].replace('%', '')
            win_rate = float(pct_str) / 100.0
            wins = int(round(battles * win_rate))
            
            print(f"   [Recovered] Bat: {battles}, Wins: {wins} ({win_rate:.1%})")
            return battles, wins
    except Exception as e: 
        print(f"⚠️ Error reading log: {e}")
        return 0, 0

def get_epsilon(battles):
    if battles >= DECAY_BATTLES: return EPS_END
    return max(EPS_END, EPS_START - (battles / DECAY_BATTLES) * (EPS_START - EPS_END))

def main():
    opponent = "random" 
    if len(sys.argv) > 1: opponent = sys.argv[1]
    
    log_file = f"v13_logs/tabular_log_{opponent}.csv"
    
    print(f"🚀 STARTING V13 TABULAR vs {opponent.upper()}")
    
    while True:
        battles, wins = get_last_stats(log_file)
        if battles >= TOTAL_BATTLES: break
            
        eps = get_epsilon(battles)
        
        print(f"\n--- Launching (Bat {battles}, Eps {eps:.3f}) ---")
        
        cmd = [
            sys.executable, "train_v13.py",
            "--historic_battles", str(battles),
            "--historic_wins", str(wins),
            "--batch_size", str(BATCH_SIZE),
            "--epsilon", str(eps),
            "--opponent", opponent
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            print("⚠️ Crash. Restarting...")
            time.sleep(1)
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()