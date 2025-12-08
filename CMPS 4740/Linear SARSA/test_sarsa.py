import asyncio
import os
import glob
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import deque

from poke_env.player import RandomPlayer, MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from sarsa_player import LinearSARSAPlayer

# --- CONFIGURATION ---
TEST_EPISODES = 1000   # How many battles to play in this session
LOG_INTERVAL = 50      # Print stats every 50 episodes
MODEL_DIR = "models"
TEST_RESULTS_DIR = "tests"

# 🤫 Silence Logs
logging.getLogger("poke_env").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

def get_latest_model():
    """Finds the most recently modified .pkl file in the models directory."""
    list_of_files = glob.glob(os.path.join(MODEL_DIR, '*.pkl'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def silence_player(player):
    player.logger.setLevel(logging.ERROR)

async def main():
    # 1. SETUP ENV
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
    
    # 2. FIND MODEL
    latest_model = get_latest_model()
    if not latest_model:
        print("❌ No model found! Run train_sarsa.py first.")
        return
    
    print(f"--- STARTING FINE-TUNING RUN ---")
    print(f"Loading Weights: {latest_model}")
    print(f"Mode: Learning Enabled (Alpha=0.0001, Epsilon=0.05)")
    
    # 3. SETUP PLAYERS
    # We enable learning (alpha > 0) and slight exploration (epsilon > 0)
    learner = LinearSARSAPlayer(
        battle_format="gen1randombattle",
        server_configuration=LocalhostServerConfiguration,
        epsilon=0.05,     # 5% Randomness (Low exploration for fine-tuning)
        alpha=0.0001,     # Learning Enabled
        gamma=0.99,
        max_concurrent_battles=1
    )
    silence_player(learner)
    
    # Load the trained brain
    learner.load_model(latest_model)
    
    # OPPONENT: Train against the "Hard" bot
    opponent = MaxBasePowerPlayer(
        battle_format="gen1randombattle",
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=1
    )
    silence_player(opponent)

    print(f"Fine-tuning against MaxBasePowerPlayer for {TEST_EPISODES} episodes...")

    # 4. TRAINING LOOP
    history_wins = []       # 1 or 0 for every battle
    rolling_window = deque(maxlen=100) # For rolling average
    
    start_time = time.time()
    
    for i in range(1, TEST_EPISODES + 1):
        await learner.battle_against(opponent, n_battles=1)
        
        # Check result
        won = 1 if learner.n_won_battles == sum(history_wins) + 1 else 0
        history_wins.append(won)
        rolling_window.append(won)
        
        if i % LOG_INTERVAL == 0:
            rolling_wr = sum(rolling_window) / len(rolling_window)
            overall_wr = sum(history_wins) / len(history_wins)
            elapsed = time.time() - start_time
            print(f"Tune {i}: Rolling {rolling_wr:.1%} | Overall {overall_wr:.1%} | Time {elapsed:.1f}s")
            
            # Save periodically in case we crash
            learner.save_model(latest_model)

    # 5. SAVING & PLOTTING
    print("\n--- FINE-TUNING FINISHED ---")
    print(f"Final Win Rate: {sum(history_wins)/TEST_EPISODES:.2%}")
    print(f"Updated weights saved to: {latest_model}")
    
    # Save Final State
    learner.save_model(latest_model)
    
    # Calculate Rolling Average for Plotting
    window_size = 50
    moving_avg = np.convolve(history_wins, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(10, 6))
    
    # Plot raw moving average
    plt.plot(range(window_size, len(history_wins)+1), moving_avg, label=f'Rolling Win Rate ({window_size} avg)', color='purple')
    
    # Plot Overall Average line
    final_wr = sum(history_wins)/TEST_EPISODES
    plt.axhline(y=final_wr, color='blue', linestyle='--', label=f'Session Average ({final_wr:.1%})')
    
    plt.title(f"Fine-Tuning Results (Learning ON)\nModel: {os.path.basename(latest_model)}")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    
    # Save Plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{TEST_RESULTS_DIR}/finetune_{timestamp}.png"
    plt.savefig(plot_path)
    print(f"Graph saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())