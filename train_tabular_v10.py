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
from tabular_player_v10 import TabularQPlayerV10

# --- CONFIG ---
BATTLES_PER_LOG = 1000
SAVE_FREQ = 1000
BATTLE_TIMEOUT = 5

# TD(Lambda) Params
ALPHA = 0.1 
GAMMA = 0.99
LAMBDA = 0.85  # <--- New Parameter: Trace Decay

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
    print(f"--- V10 TABULAR TD(λ={LAMBDA}) WORKER ---")
    run_uuid = uuid.uuid4().hex[:8]
    
    if args.opponent == "random": BaseOpp = RandomPlayer
    elif args.opponent == "maxbp": BaseOpp = MaxBasePowerPlayer
    else: BaseOpp = SimpleHeuristicsPlayer
    
    OppClass = get_unique_player_class(BaseOpp, "Opp", run_uuid)
    opponent = OppClass(battle_format="gen1randombattle", 
                        server_configuration=LocalhostServerConfiguration, 
                        max_concurrent_battles=1)

    LearnerClass = get_unique_player_class(TabularQPlayerV10, "Learner", run_uuid)
    learner = LearnerClass(battle_format="gen1randombattle", 
                           server_configuration=LocalhostServerConfiguration,
                           max_concurrent_battles=1,
                           alpha=ALPHA, gamma=GAMMA, lam=LAMBDA, epsilon=args.epsilon)
    
    MODEL_FILE = f"v10_models/qtable_{args.opponent}.pkl"
    os.makedirs("v10_models", exist_ok=True)
    os.makedirs("v10_logs", exist_ok=True)
    
    if os.path.exists(MODEL_FILE):
        learner.load_table(MODEL_FILE)

    learner.logger.setLevel(logging.CRITICAL)
    opponent.logger.setLevel(logging.CRITICAL)

    battles_collected = 0
    start_time = time.time()
    
    session_wins = 0
    last_total_wins = learner.n_won_battles
    next_log = BATTLES_PER_LOG

    while battles_collected < args.batch_size:
        try:
            await asyncio.wait_for(learner.battle_against(opponent, n_battles=1), timeout=BATTLE_TIMEOUT)
            
            won = 1 if learner.n_won_battles > last_total_wins else 0
            last_total_wins = learner.n_won_battles
            
            session_wins += won
            battles_collected += 1
            
            total_battles = args.historic_battles + battles_collected
            
            if battles_collected >= next_log:
                elapsed = time.time() - start_time
                speed = battles_collected / elapsed if elapsed > 0 else 0.0
                
                total_wins_all = args.historic_wins + session_wins
                overall_wr = total_wins_all / total_battles
                rolling_wr = session_wins / battles_collected
                table_size = len(learner.q_table)
                
                print(f"Bat {total_battles}: RollingWin {rolling_wr:.1%} | Overall {overall_wr:.1%} | Eps {learner.epsilon:.3f} | States {table_size} | Speed {speed:.1f}/s")
                
                log_stats(
                    f"v10_logs/tabular_log_{args.opponent}.csv",
                    total_battles, rolling_wr, overall_wr, learner.epsilon, speed, table_size, args.opponent
                )
                next_log += BATTLES_PER_LOG
            
            if total_battles % SAVE_FREQ == 0:
                learner.save_table(MODEL_FILE)

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