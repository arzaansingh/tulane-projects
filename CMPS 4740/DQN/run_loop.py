import subprocess
import sys
import time
import os
import csv

# --- CONFIG ---
TOTAL_EPISODES = 1000000
BATCH_SIZE = 10000 # Restart Python every 1000 battles to clear memory
LOG_INTERVAL = 1000

# Epsilon Schedule
DECAY_STEPS = 100000
EPS_START = 0.5
EPS_END = 0.01

def get_last_stats(log_file):
    current_ep = 0
    historic_wins = 0
    
    if not os.path.exists(log_file): return 0, 0
    try:
        with open(log_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) > 1:
                last_row = data[-1]
                # Format: Episode, RollingWin, OverallWin, Epsilon, Speed, Opponent
                if len(last_row) >= 3:
                    current_ep = int(last_row[0])
                    overall_wr = float(last_row[2])
                    historic_wins = int(current_ep * overall_wr)
    except: pass 
    return current_ep, historic_wins

def get_epsilon(current_ep):
    if current_ep >= DECAY_STEPS: return EPS_END
    prog = current_ep / DECAY_STEPS
    return max(EPS_END, EPS_START - (prog * (EPS_START - EPS_END)))

def main():
    # Default to heuristic, or take from command line
    opponent = "heuristic"
    if len(sys.argv) > 1:
        opponent = sys.argv[1]

    log_file = f"v4_logs/dqn_log_{opponent}.csv"
    
    print(f"🚀 STARTING DQN TRAINING vs {opponent.upper()}")
    print(f"   Device: Auto-detecting (likely MPS/CPU)")
    print(f"   Goal: {TOTAL_EPISODES} episodes")
    
    while True:
        current_ep, historic_wins = get_last_stats(log_file)
        
        if current_ep >= TOTAL_EPISODES:
            print("🎉 Target Reached.")
            break
            
        eps = get_epsilon(current_ep)
        
        print(f"\n--- Launching Worker (Start: {current_ep}, Eps: {eps:.4f}) ---")
        
        cmd = [
            sys.executable, "train_dqn.py",
            "--start_ep", str(current_ep),
            "--batch_size", str(BATCH_SIZE),
            "--historic_wins", str(historic_wins),
            "--epsilon", str(eps),
            "--opponent", opponent
        ]
        
        # Run worker and wait for it to finish/die
        p = subprocess.run(cmd)
        
        if p.returncode != 0:
            print("⚠️ Worker crashed. Restarting in 5s...")
            time.sleep(5)
        else:
            # Brief pause to ensure file handles close
            time.sleep(1) 

if __name__ == "__main__":
    main()