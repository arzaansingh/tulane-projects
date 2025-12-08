import asyncio
import os
import csv
import time
import logging
import uuid
import sys
import argparse
from collections import deque
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer, MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from dqn_player import DQNPlayer

# Config
BATCH_SIZE = 10000 # Train network after every 1000 battles (or less)
BATTLE_TIMEOUT = 10
LOG_INTERVAL = 1000
TARGET_UPDATE_FREQ = 100 # Update target network every 5k steps

logging.getLogger("poke_env").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

os.makedirs("v4_logs", exist_ok=True)
os.makedirs("v4_models", exist_ok=True)

def log_stats(filename, episode, rolling_win, overall_win, epsilon, speed, opponent):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'RollingWin', 'OverallWin', 'Epsilon', 'Speed', 'Opponent'])
        writer.writerow([episode, rolling_win, overall_win, epsilon, speed, opponent])

def get_unique_player_class(base_class, prefix, run_uuid):
    unique_name = f"{prefix}_{run_uuid}"
    return type(unique_name, (base_class,), {})

async def main(args):
    run_uuid = uuid.uuid4().hex[:8]
    
    if args.opponent == "random": BaseOpponent = RandomPlayer
    elif args.opponent == "maxbp": BaseOpponent = MaxBasePowerPlayer
    else: BaseOpponent = SimpleHeuristicsPlayer

    OpponentClass = get_unique_player_class(BaseOpponent, "Opp", run_uuid)
    opponent = OpponentClass(battle_format="gen1randombattle", server_configuration=LocalhostServerConfiguration, max_concurrent_battles=1)
    opponent.logger.setLevel(logging.ERROR)

    MODEL_FILE = f"v4_models/dqn_{args.opponent}.pth"
    LOG_FILE = f"v4_logs/dqn_log_{args.opponent}.csv"

    LearnerClass = get_unique_player_class(DQNPlayer, "DQN", run_uuid)
    learner = LearnerClass(
        battle_format="gen1randombattle",
        server_configuration=LocalhostServerConfiguration,
        epsilon=args.epsilon,
        max_concurrent_battles=1
    )
    learner.logger.setLevel(logging.ERROR)
    
    if os.path.exists(MODEL_FILE):
        learner.load_checkpoint(MODEL_FILE)

    print(f"--- DQN TRAINING START: {args.start_ep} | Eps: {args.epsilon:.4f} ---")
    
    # Track wins within THIS worker session
    session_wins = 0          # total wins in this batch
    prev_wins = 0             # wins at last LOG_INTERVAL snapshot

    # Baseline for poke-env's cumulative win counter
    initial_learner_wins = learner.n_won_battles
    last_total_wins = initial_learner_wins
    
    battles_done = 0
    start_time = time.time()
    
    # Training Loop
    while battles_done < args.batch_size:
        try:
            await asyncio.wait_for(learner.battle_against(opponent, n_battles=1), timeout=BATTLE_TIMEOUT)
            
            # Detect win using the monotone n_won_battles counter
            current_total_wins = learner.n_won_battles
            if current_total_wins > last_total_wins:
                won = 1
            else:
                won = 0
            last_total_wins = current_total_wins

            session_wins += won
            battles_done += 1
            current_total = args.start_ep + battles_done

            # Train the network
            # We do a few optimization steps per battle to learn faster
            for _ in range(5):
                learner.optimize_model()

            # Update Target Network
            if current_total % TARGET_UPDATE_FREQ == 0:
                learner.update_target_net()

            # Log
            if current_total % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                speed = battles_done / elapsed if elapsed > 0 else 0.0

                # Rolling win rate over the last LOG_INTERVAL battles
                wins_this_interval = session_wins - prev_wins
                rolling_win = wins_this_interval / LOG_INTERVAL if LOG_INTERVAL > 0 else 0.0
                prev_wins = session_wins

                # Overall win rate across ALL battles so far
                overall_wins = args.historic_wins + session_wins
                overall_win = overall_wins / current_total if current_total > 0 else 0.0
                
                print(f"Ep {current_total}: RollWin {rolling_win:.2%} | Overall {overall_win:.2%} | Eps {args.epsilon:.3f} | Speed {speed:.1f}")
                log_stats(LOG_FILE, current_total, rolling_win, overall_win, args.epsilon, speed, args.opponent)
                
                learner.save_checkpoint(MODEL_FILE)

        except asyncio.TimeoutError: pass
        except Exception: pass
            
    learner.save_checkpoint(MODEL_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--historic_wins", type=int, default=0) 
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--opponent", type=str, default="heuristic")
    args = parser.parse_args()
    asyncio.run(main(args))