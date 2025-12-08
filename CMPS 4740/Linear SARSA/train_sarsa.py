import asyncio
import os
import csv
import time
import logging
from collections import deque
from datetime import datetime
from poke_env.player import MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from sarsa_player import LinearSARSAPlayer

# --- CONFIGURATION ---
TOTAL_EPISODES = 1000000   
SAVE_INTERVAL = 2000       
LOG_INTERVAL = 1000        # Log exactly every 1000 battles
MAX_CONCURRENT = 10        # Number of parallel workers
BATTLE_TIMEOUT = 10        # Kill battle if it takes > 10s (prevents hanging)
VERBOSE = False            

TRAIN_NEW_MODEL = False    

# 🤫 SILENCE GLOBAL LOGS
logging.getLogger("poke_env").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

if TRAIN_NEW_MODEL:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"logs/training_log_hard_{run_id}.csv"
    MODEL_FILE = f"models/sarsa_weights_hard_{run_id}.pkl"
    print(f"--- STARTING NEW RUN: {run_id} ---")
else:
    LOG_FILE = "logs/training_log_master.csv"
    MODEL_FILE = "models/sarsa_master.pkl"
    print(f"--- CONTINUING MASTER RUN ---")

def get_start_episode():
    """Reads the last episode number from the log file to resume counting."""
    if not os.path.exists(LOG_FILE):
        return 0
    try:
        with open(LOG_FILE, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) > 1: 
                return int(data[-1][0])
    except Exception as e:
        print(f"⚠️ Could not read previous episode count: {e}")
    return 0

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
    def __init__(self, start_ep):
        self.battles_done = 0
        self.start_episode = start_ep
        self.recent_outcomes = deque(maxlen=1000) # Rolling avg over 1000
        self.start_time = time.time()
        # Set the next target (e.g. 1000, 2000...)
        self.next_log_target = LOG_INTERVAL
        self.next_save_target = SAVE_INTERVAL

async def battle_worker(worker_id, learner, opponent, state):
    """
    Independent worker that keeps playing battles until the goal is reached.
    """
    while state.battles_done < TOTAL_EPISODES:
        # 1. Run ONE battle
        wins_before = learner.n_won_battles
        try:
            # Enforce timeout to prevent stuck workers
            await asyncio.wait_for(
                learner.battle_against(opponent, n_battles=1),
                timeout=BATTLE_TIMEOUT 
            )
        except asyncio.TimeoutError:
            # If timed out, just continue to next battle
            # print(f"⚠️ Worker {worker_id} timed out (> {BATTLE_TIMEOUT}s). Skipping battle...")
            pass
        except Exception as e:
            print(f"⚠️ Worker {worker_id} error: {e}")

        # 2. Update Shared State
        wins_after = learner.n_won_battles
        did_win = 1 if wins_after > wins_before else 0
        
        state.recent_outcomes.append(did_win)
        state.battles_done += 1
        
        if learner.tau > 0.1:
            learner.tau *= 0.99995

        # 3. Precise Logging & Saving
        # Check if we just crossed the threshold
        if state.battles_done >= state.next_log_target:
            # Advance target immediately to prevent double-logging
            state.next_log_target += LOG_INTERVAL
            
            current_total = state.start_episode + state.battles_done
            
            rolling_wr = sum(state.recent_outcomes) / len(state.recent_outcomes) if state.recent_outcomes else 0.0
            session_wr = learner.n_won_battles / state.battles_done if state.battles_done > 0 else 0.0
            
            elapsed = time.time() - state.start_time
            s_per_battle = elapsed / state.battles_done if state.battles_done > 0 else 0
            
            print(f"Ep {current_total}: Rolling {rolling_wr:.2%} | Session {session_wr:.2%} | Tau {learner.tau:.2f} | Speed {1/s_per_battle:.1f} bat/s")
            log_stats(current_total, session_wr, learner.tau, "MaxBasePower")

        if state.battles_done >= state.next_save_target:
            state.next_save_target += SAVE_INTERVAL
            learner.save_model(MODEL_FILE)

async def main():
    FORMAT = "gen1randombattle"
    
    # Initialize Learner
    learner = LinearSARSAPlayer(
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
        tau=5.0,     
        alpha=0.01, 
        gamma=0.99,
        max_concurrent_battles=MAX_CONCURRENT
    )
    silence_player(learner)
    
    # Resume Logic
    start_episode = 0
    if not TRAIN_NEW_MODEL:
        if os.path.exists(MODEL_FILE):
            learner.load_model(MODEL_FILE)
        start_episode = get_start_episode()
        print(f"🔄 Resuming from Episode {start_episode}...")
    
    # Print Weights for debugging
    print("\n" + "="*60)
    print(f"📊 INITIAL FEATURE WEIGHTS (Total: {len(learner.weights)})")
    for i, w in enumerate(learner.weights[:20]):
        print(f"Feat {i:02d}: {w:+.5f}", end="\t")
        if (i + 1) % 4 == 0: print()
    print("\n... (showing first 20)\n" + "="*60 + "\n")

    # Initialize Opponent
    opponent = MaxBasePowerPlayer(
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=MAX_CONCURRENT
    )
    silence_player(opponent)

    print(f"🚀 Launching {MAX_CONCURRENT} parallel workers...")
    
    state = TrainingState(start_episode)
    
    tasks = []
    for i in range(MAX_CONCURRENT):
        task = asyncio.create_task(battle_worker(i, learner, opponent, state))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

    print("Training finished.")
    learner.save_model(MODEL_FILE)

if __name__ == "__main__":
    asyncio.run(main())