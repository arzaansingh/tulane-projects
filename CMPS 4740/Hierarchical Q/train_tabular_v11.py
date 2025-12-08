import asyncio
import os
import csv
import time
import logging
import uuid
import sys
import argparse
import traceback
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer, MaxBasePowerPlayer
from tabular_player_v11 import TabularQPlayerV11

# --- CONFIG ---
BATTLES_PER_LOG = 1000 
SAVE_FREQ = 100
BATTLE_TIMEOUT = 10

ALPHA = 0.1 
GAMMA = 0.995 
LAMBDA = 0.6967 # V11 uses dynamic action space, so traces might be sparser, 0.8 is safe

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)

def log_stats(filename, battles, rolling_win, overall_win, epsilon, speed, table_size, opponent):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Battles', 'RollingWin', 'OverallWin', 'Epsilon', 'Speed', 'TableSize', 'Opponent'])
        writer.writerow([battles, f"{rolling_win:.2%}", f"{overall_win:.2%}", f"{epsilon:.4f}", speed, table_size, opponent])

def get_unique_player_class(base_class, prefix, run_uuid):
    return type(f"{prefix}_{run_uuid}", (base_class,), {})

async def main(args):
    print(f"--- V11 TABULAR WORKER (Bat: {args.historic_battles}) ---")
    run_uuid = uuid.uuid4().hex[:8]
    
    if args.opponent == "random": BaseOpp = RandomPlayer
    elif args.opponent == "maxbp": BaseOpp = MaxBasePowerPlayer
    else: BaseOpp = SimpleHeuristicsPlayer
    
    OppClass = get_unique_player_class(BaseOpp, "Opp", run_uuid)
    opponent = OppClass(battle_format="gen1randombattle", 
                        server_configuration=LocalhostServerConfiguration, 
                        max_concurrent_battles=1)

    LearnerClass = get_unique_player_class(TabularQPlayerV11, "Learner", run_uuid)
    learner = LearnerClass(battle_format="gen1randombattle", 
                           server_configuration=LocalhostServerConfiguration,
                           max_concurrent_battles=1,
                           alpha=ALPHA, gamma=GAMMA, lam=LAMBDA, epsilon=args.epsilon)
    
    MODEL_FILE = f"v11_models/qtable_{args.opponent}.pkl"
    os.makedirs("v11_models", exist_ok=True)
    os.makedirs("v11_logs", exist_ok=True)
    
    if os.path.exists(MODEL_FILE):
        learner.load_table(MODEL_FILE)

    learner.logger.setLevel(logging.CRITICAL)
    opponent.logger.setLevel(logging.CRITICAL)

    battles_collected = 0
    start_time = time.time()
    
    total_battles_processed = args.historic_battles
    total_wins_overall = args.historic_wins
    
    session_wins = 0
    last_total_wins = learner.n_won_battles
    
    next_log = BATTLES_PER_LOG

    while battles_collected < args.batch_size:
        try:
            if battles_collected > 0 and battles_collected % SAVE_FREQ == 0:
                 learner.save_table(MODEL_FILE)
            
            await asyncio.wait_for(learner.battle_against(opponent, n_battles=1), timeout=BATTLE_TIMEOUT)
            
            current_p1_wins = learner.n_won_battles
            won = 1 if current_p1_wins > last_total_wins else 0
            last_total_wins = current_p1_wins
            
            session_wins += won
            battles_collected += 1
            
            total_battles_processed = args.historic_battles + battles_collected
            total_wins_overall = args.historic_wins + session_wins

            if battles_collected >= next_log:
                elapsed = time.time() - start_time
                speed = battles_collected / elapsed if elapsed > 0 else 0.0
                
                overall_wr = total_wins_overall / total_battles_processed
                rolling_wr = session_wins / battles_collected
                table_size = len(learner.q_table) # Counts (state, action) pairs
                
                print(f"Bat {total_battles_processed}: Win {rolling_wr:.0%} | Overall {overall_wr:.0%} | Eps {learner.epsilon:.3f} | States {table_size} | Speed {speed:.1f}/s")
                
                log_stats(
                    f"v11_logs/tabular_log_{args.opponent}.csv",
                    total_battles_processed, rolling_wr, overall_wr, learner.epsilon, speed, table_size, args.opponent
                )
                next_log += BATTLES_PER_LOG

        except Exception as e:
            pass

    learner.save_table(MODEL_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historic_battles", type=int, default=0)
    parser.add_argument("--historic_wins", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--opponent", type=str, default="random")
    args = parser.parse_args()
    asyncio.run(main(args))