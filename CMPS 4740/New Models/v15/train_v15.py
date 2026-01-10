import asyncio
import os
import csv
import time
import logging
import uuid
import sys
import argparse
import traceback
from collections import deque
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer, MaxBasePowerPlayer
from player_v15 import TabularQPlayerV15

# --- CONFIG ---
BATTLES_PER_LOG = 1000 
SAVE_FREQ = 1000
BATTLE_TIMEOUT = 1 

ALPHA = 0.1 
GAMMA = 0.995 
LAMBDA = 0.6967 

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)

def log_stats(filename, battles, rolling_win, overall_win, epsilon, speed, avg_rew, table_size, opponent):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Battles', 'RollingWin', 'OverallWin', 'Epsilon', 'Speed', 'AvgReward', 'TableSize', 'Opponent'])
        writer.writerow([battles, f"{rolling_win:.2%}", f"{overall_win:.2%}", f"{epsilon:.4f}", speed, f"{avg_rew:.4f}", table_size, opponent])

def print_live_progress(current_count, total_count, current_speed, log_window_size):
    progress_fraction = current_count / log_window_size
    progress_fraction = min(progress_fraction, 1.0)
    bar_len = 20
    filled = int(progress_fraction * bar_len)
    bar = f"[{'█' * filled}{'-' * (bar_len - filled)}]"
    sys.stdout.write(f"\rProgress {bar} {progress_fraction:.0%} (Bat {current_count}/{log_window_size}, Speed {current_speed:.1f}/s)")
    sys.stdout.flush()

def get_unique_player_class(base_class, prefix, run_uuid):
    return type(f"{prefix}_{run_uuid}", (base_class,), {})

async def main(args):
    if args.historic_battles == 0:
        print(f"--- V15 TABULAR WORKER (Target Batch: {args.batch_size}) ---")
    
    run_uuid = uuid.uuid4().hex[:8]
    
    if args.opponent == "random": BaseOpp = RandomPlayer
    elif args.opponent == "maxbp": BaseOpp = MaxBasePowerPlayer
    else: BaseOpp = SimpleHeuristicsPlayer
    
    OppClass = get_unique_player_class(BaseOpp, "Opp", run_uuid)
    opponent = OppClass(battle_format="gen1randombattle", 
                        server_configuration=LocalhostServerConfiguration, 
                        max_concurrent_battles=1)

    LearnerClass = get_unique_player_class(TabularQPlayerV15, "Learner", run_uuid)
    learner = LearnerClass(battle_format="gen1randombattle", 
                           server_configuration=LocalhostServerConfiguration,
                           max_concurrent_battles=1,
                           alpha=ALPHA, gamma=GAMMA, lam=LAMBDA, epsilon=args.epsilon)
    
    MODEL_FILE = f"v15_models/qtable_{args.opponent}.pkl"
    os.makedirs("v15_models", exist_ok=True)
    os.makedirs("v15_logs", exist_ok=True)
    
    if os.path.exists(MODEL_FILE):
        learner.load_table(MODEL_FILE)

    battles_collected = 0
    start_time = time.time()
    
    initial_learner_wins = learner.n_won_battles 
    session_outcomes = deque(maxlen=BATTLES_PER_LOG)
    
    accumulated_total_reward = 0.0 
    
    chunk_size = 1 
    log_window_start_time = time.time()
    current_log_progress = 0
    consecutive_timeouts = 0
    
    while battles_collected < args.batch_size:
        try:
            if battles_collected > 0 and battles_collected % SAVE_FREQ == 0:
                 learner.save_table(MODEL_FILE)

            wins_before = learner.n_won_battles
            
            await asyncio.wait_for(learner.battle_against(opponent, n_battles=chunk_size), timeout=BATTLE_TIMEOUT)
            
            consecutive_timeouts = 0
            
            is_win = learner.n_won_battles > wins_before
            session_outcomes.append(1 if is_win else 0)
            
            outcome_reward = 1.0 if is_win else -1.0
            accumulated_total_reward += outcome_reward
            
            step_rewards = learner.pop_step_rewards()
            accumulated_total_reward += sum(step_rewards)
            
            battles_collected += chunk_size
            current_log_progress += chunk_size
            
            if current_log_progress % 10 == 0 and current_log_progress < BATTLES_PER_LOG:
                elapsed_progress = time.time() - log_window_start_time
                current_speed = current_log_progress / elapsed_progress if elapsed_progress > 0 else 0
                print_live_progress(current_log_progress, BATTLES_PER_LOG, current_speed, BATTLES_PER_LOG)
            
            if battles_collected % BATTLES_PER_LOG == 0:
                sys.stdout.write("\r" + " " * 80 + "\r")
                sys.stdout.flush()
                
                rolling_wr = sum(session_outcomes) / len(session_outcomes) if len(session_outcomes) > 0 else 0.0
                current_session_wins = learner.n_won_battles - initial_learner_wins
                total_battles_processed = args.historic_battles + battles_collected
                total_wins_overall = args.historic_wins + current_session_wins
                overall_wr = total_wins_overall / total_battles_processed if total_battles_processed > 0 else 0.0
                
                elapsed = time.time() - start_time
                speed = battles_collected / elapsed if elapsed > 0 else 0.0
                table_size = len(learner.q_table)
                avg_rew = accumulated_total_reward / BATTLES_PER_LOG
                
                print(f"Bat {total_battles_processed}: Rolling {rolling_wr:.2%} | Overall {overall_wr:.2%} | AvgRew {avg_rew:.3f} | Eps {learner.epsilon:.3f} | States {table_size} | Speed {speed:.1f}/s")
                
                log_stats(
                    f"v15_logs/log_{args.opponent}.csv",
                    total_battles_processed, rolling_wr, overall_wr, learner.epsilon, speed, avg_rew, table_size, args.opponent
                )
                
                accumulated_total_reward = 0.0
                current_log_progress = 0
                log_window_start_time = time.time()

        except asyncio.TimeoutError:
            consecutive_timeouts += 1
            if consecutive_timeouts >= 5:
                print(f"\n⚠️ 5 Timeouts. Restarting Process.")
                learner.save_table(MODEL_FILE)
                sys.exit(1) 
            time.sleep(0.1)
            continue
        except Exception:
            traceback.print_exc()
            pass

    learner.save_table(MODEL_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historic_battles", type=int, default=0)
    parser.add_argument("--historic_wins", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--opponent", type=str, default="maxbp")
    args = parser.parse_args()
    asyncio.run(main(args))