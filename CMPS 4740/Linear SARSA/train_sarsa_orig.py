import asyncio
import os
import csv
import time
import logging
from collections import deque
from datetime import datetime
from poke_env.player import MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
# Import the ORIG player
from sarsa_player_orig import LinearSARSAPlayer

# --- CONFIGURATION ---
TOTAL_EPISODES = 1000000   
SAVE_INTERVAL = 2000       
LOG_INTERVAL = 1000        
MAX_CONCURRENT = 10        
BATTLE_TIMEOUT = 60        
VERBOSE = False            

TRAIN_NEW_MODEL = False    

# 🤫 SILENCE GLOBAL LOGS
logging.getLogger("poke_env").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

if TRAIN_NEW_MODEL:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"logs/training_log_orig_{run_id}.csv"
    MODEL_FILE = f"models/sarsa_weights_orig_{run_id}.pkl"
    print(f"--- STARTING NEW RUN (ORIG): {run_id} ---")
else:
    LOG_FILE = "logs/training_log_master_orig.csv"
    MODEL_FILE = "models/sarsa_master_orig.pkl"
    print(f"--- CONTINUING MASTER RUN (ORIG) ---")

def get_start_stats():
    """Reads the last episode and win rate to calculate historic wins."""
    start_ep = 0
    start_wins = 0
    
    if not os.path.exists(LOG_FILE): 
        return 0, 0
        
    try:
        with open(LOG_FILE, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            # Check if file has content and header
            if len(data) > 1: 
                last_row = data[-1]
                # Column 0: Episode, Column 1: WinRate (0.0 to 1.0)
                if len(last_row) >= 2:
                    start_ep = int(last_row[0])
                    win_rate = float(last_row[1])
                    # Reverse calculate total wins from the percentage
                    start_wins = int(start_ep * win_rate)
    except Exception as e:
        print(f"⚠️ Could not read previous stats: {e}")
        
    return start_ep, start_wins

def log_stats(episode, win_rate, tau, opponent_name):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'WinRate', 'Tau', 'Opponent'])
        writer.writerow([episode, win_rate, tau, opponent_name])

def silence_player(player):
    if not VERBOSE:
        player.logger.setLevel(logging.ERROR)

# --- SHARED STATE FOR WORKERS ---
class TrainingState:
    def __init__(self, start_ep, start_wins):
        self.battles_done = 0
        self.start_episode = start_ep
        self.start_wins = start_wins
        
        self.start_time = time.time()
        self.next_log_target = LOG_INTERVAL
        self.next_save_target = SAVE_INTERVAL
        
        # Trackers for rolling calculation
        self.last_log_battles = 0
        self.last_log_wins = 0

async def battle_worker(worker_id, learner, opponent, state):
    while state.battles_done < TOTAL_EPISODES:
        try:
            await asyncio.wait_for(
                learner.battle_against(opponent, n_battles=1),
                timeout=BATTLE_TIMEOUT 
            )
        except asyncio.TimeoutError:
            print(f"⚠️ Worker {worker_id} timed out. Skipping...")
            pass
        except Exception as e:
            pass

        # Increment global counter
        state.battles_done += 1
        
        if learner.tau > 0.1:
            learner.tau *= 0.9995

        # Logging & Saving Logic
        # We check if we've crossed the log threshold
        if state.battles_done >= state.next_log_target:
            # Use a small window check to ensure only one worker triggers the print
            if state.battles_done < state.next_log_target + MAX_CONCURRENT:
                state.next_log_target += LOG_INTERVAL
                
                # --- SNAPSHOT STATS ---
                current_session_wins = learner.n_won_battles
                current_session_battles = state.battles_done
                
                # 1. Rolling Win Rate (Since last log)
                delta_battles = current_session_battles - state.last_log_battles
                delta_wins = current_session_wins - state.last_log_wins
                rolling_wr = delta_wins / delta_battles if delta_battles > 0 else 0.0
                
                # 2. Overall Cumulative Win Rate (Historic + Session)
                total_battles = state.start_episode + current_session_battles
                total_wins = state.start_wins + current_session_wins
                cumulative_wr = total_wins / total_battles if total_battles > 0 else 0.0
                
                # Update Snapshot
                state.last_log_battles = current_session_battles
                state.last_log_wins = current_session_wins
                
                elapsed = time.time() - state.start_time
                s_per_battle = elapsed / current_session_battles if current_session_battles > 0 else 0
                
                print(f"Ep {total_battles}: Rolling {rolling_wr:.2%} | Overall {cumulative_wr:.2%} | Tau {learner.tau:.2f} | Speed {1/s_per_battle:.1f} bat/s")
                log_stats(total_battles, cumulative_wr, learner.tau, "MaxBasePower")

        if state.battles_done >= state.next_save_target:
            state.next_save_target += SAVE_INTERVAL
            learner.save_model(MODEL_FILE)

async def main():
    FORMAT = "gen1randombattle"
    
    learner = LinearSARSAPlayer(
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
        tau=1e8,     
        alpha=0.01, 
        gamma=0.99,
        max_concurrent_battles=MAX_CONCURRENT
    )
    silence_player(learner)
    
    # Initialize Start Stats
    start_episode = 0
    start_wins = 0
    
    if not TRAIN_NEW_MODEL:
        if os.path.exists(MODEL_FILE):
            learner.load_model(MODEL_FILE)
        # Load history
        start_episode, start_wins = get_start_stats()
        print(f"🔄 Resuming from Episode {start_episode} (Historic Wins: {start_wins})...")
    
    opponent = MaxBasePowerPlayer(
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=MAX_CONCURRENT
    )
    silence_player(opponent)

    print(f"🚀 Launching {MAX_CONCURRENT} parallel workers...")
    
    # Initialize State with historic data
    state = TrainingState(start_episode, start_wins)
    
    tasks = []
    for i in range(MAX_CONCURRENT):
        task = asyncio.create_task(battle_worker(i, learner, opponent, state))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

    print("Training finished.")
    learner.save_model(MODEL_FILE)

if __name__ == "__main__":
    asyncio.run(main())