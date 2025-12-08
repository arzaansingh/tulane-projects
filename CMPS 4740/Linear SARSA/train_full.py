import asyncio
import os
import csv
import time
import logging
from collections import deque
from datetime import datetime
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
# Import the FULL player
from sarsa_player_full import LinearSARSAPlayer

# --- CONFIGURATION ---
TOTAL_EPISODES = 1000000   
SAVE_INTERVAL = 2000       
LOG_INTERVAL = 1000        
MAX_CONCURRENT = 10        
BATTLE_TIMEOUT = 60        
VERBOSE = False            

TRAIN_NEW_MODEL = False    

# --- EXPLORATION SCHEDULE ---
# Epsilon: Linear 1.0 -> 0.0 over first 5% of battles
EPS_DECAY_STEPS = 50000
EPS_START = 1.0
EPS_END = 0.0

# Tau: Exponential 1e9 -> ~0.0 (Unbounded Decay)
# We calculate rate to hit 0.1 at 500k, but let it keep going down forever
TAU_START = 1e8
TAU_TARGET_AT_HALF = 0.1 # Target value at 500k steps
# decay = (target / start) ^ (1 / steps)
TAU_DECAY_RATE = (TAU_TARGET_AT_HALF / TAU_START) ** (1 / EPS_DECAY_STEPS)

# 🤫 SILENCE GLOBAL LOGS
logging.getLogger("poke_env").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

if TRAIN_NEW_MODEL:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = f"logs/training_log_full_{run_id}.csv"
    MODEL_FILE = f"models/sarsa_weights_full_{run_id}.pkl"
    print(f"--- STARTING NEW RUN (FULL): {run_id} ---")
else:
    LOG_FILE = "logs/training_log_master_full.csv"
    MODEL_FILE = "models/sarsa_master_full.pkl"
    print(f"--- CONTINUING MASTER RUN (FULL) ---")

def get_start_stats():
    start_ep = 0
    start_wins = 0
    if not os.path.exists(LOG_FILE): return 0, 0
    try:
        with open(LOG_FILE, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            if len(data) > 1: 
                last_row = data[-1]
                if len(last_row) >= 2:
                    start_ep = int(last_row[0])
                    win_rate = float(last_row[1])
                    start_wins = int(start_ep * win_rate)
    except: pass
    return start_ep, start_wins

def log_stats(episode, win_rate, tau, epsilon, opponent_name):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'WinRate', 'Tau', 'Epsilon', 'Opponent'])
        writer.writerow([episode, win_rate, tau, epsilon, opponent_name])

def silence_player(player):
    if not VERBOSE:
        player.logger.setLevel(logging.ERROR)

class TrainingState:
    def __init__(self, start_ep, start_wins):
        self.battles_done = 0
        self.start_episode = start_ep
        self.start_wins = start_wins
        self.start_time = time.time()
        self.next_log_target = LOG_INTERVAL
        self.next_save_target = SAVE_INTERVAL
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

        state.battles_done += 1
        current_total = state.start_episode + state.battles_done
        
        # --- UPDATE EXPLORATION PARAMETERS ---
        # 1. Epsilon Decay (Linear Stop at 0)
        if current_total <= EPS_DECAY_STEPS:
            progress = current_total / EPS_DECAY_STEPS
            learner.epsilon = max(EPS_END, EPS_START - (progress * (EPS_START - EPS_END)))
        else:
            learner.epsilon = 0.0
            
        # 2. Tau Decay (Exponential Unbounded)
        # We simply multiply by decay rate every step. No floor.
        # The Player class handles extremely small Taus by switching to Argmax.
        learner.tau *= TAU_DECAY_RATE

        if state.battles_done >= state.next_log_target:
            if state.battles_done < state.next_log_target + MAX_CONCURRENT:
                state.next_log_target += LOG_INTERVAL
                
                current_session_wins = learner.n_won_battles
                current_session_battles = state.battles_done
                
                delta_battles = current_session_battles - state.last_log_battles
                delta_wins = current_session_wins - state.last_log_wins
                rolling_wr = delta_wins / delta_battles if delta_battles > 0 else 0.0
                
                total_battles = state.start_episode + current_session_battles
                total_wins = state.start_wins + current_session_wins
                cumulative_wr = total_wins / total_battles if total_battles > 0 else 0.0
                
                state.last_log_battles = current_session_battles
                state.last_log_wins = current_session_wins
                
                elapsed = time.time() - state.start_time
                s_per_battle = elapsed / current_session_battles if current_session_battles > 0 else 0
                
                print(f"Ep {total_battles}: Rolling {rolling_wr:.2%} | Overall {cumulative_wr:.2%} | Tau {learner.tau:.2e} | Eps {learner.epsilon:.3f} | Speed {1/s_per_battle:.1f} bat/s")
                log_stats(total_battles, cumulative_wr, learner.tau, learner.epsilon, "SimpleHeuristics")

        if state.battles_done >= state.next_save_target:
            state.next_save_target += SAVE_INTERVAL
            learner.save_model(MODEL_FILE)

async def main():
    FORMAT = "gen1randombattle"
    
    # 1. Setup Learner (No PlayerConfiguration, let poke-env handle names)
    learner = LinearSARSAPlayer(
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
        tau=TAU_START,
        epsilon=EPS_START,
        alpha=0.001, 
        gamma=0.99,
        max_concurrent_battles=MAX_CONCURRENT
    )
    silence_player(learner)
    
    start_episode = 0
    start_wins = 0
    if not TRAIN_NEW_MODEL:
        if os.path.exists(MODEL_FILE):
            learner.load_model(MODEL_FILE)
        start_episode, start_wins = get_start_stats()
        print(f"🔄 Resuming from Episode {start_episode} (Historic Wins: {start_wins})...")
        
        # Fast-forward parameters if resuming
        if start_episode > 0:
            progress = min(1.0, start_episode / EPS_DECAY_STEPS)
            learner.epsilon = max(EPS_END, EPS_START - (progress * (EPS_START - EPS_END)))
            learner.tau = TAU_START * (TAU_DECAY_RATE ** start_episode)
            
            print(f"   Adjusted Params -> Eps: {learner.epsilon:.4f}, Tau: {learner.tau:.2e}")
    
    # 2. Setup Opponent (SimpleHeuristicsPlayer, No PlayerConfiguration)
    opponent = SimpleHeuristicsPlayer(
        battle_format=FORMAT,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=MAX_CONCURRENT
    )
    silence_player(opponent)

    print(f"🚀 Launching {MAX_CONCURRENT} parallel workers...")
    print(f"Goal: {TOTAL_EPISODES} battles against SimpleHeuristicsPlayer.")
    print(f"Decay Phase: 0 -> {EPS_DECAY_STEPS} battles.")
    
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