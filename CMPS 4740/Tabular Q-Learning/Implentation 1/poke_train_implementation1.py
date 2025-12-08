import asyncio
import numpy as np
import random
import pickle
import os

from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.battle import pokemon 

# Monkey patch to fix Gen 1 crashes and NoneType errors
_original_available_moves = pokemon.Pokemon.available_moves_from_request

def patched_available_moves(self, request_moves):
    try:
        return _original_available_moves(self, request_moves)
    except AssertionError:
        # If Gen 1 mechanics confuse the library, return an empty list 
        # instead of None. This prevents errors.
        return []

pokemon.Pokemon.available_moves_from_request = patched_available_moves

# Custom Helper to bypass the BattleOrder error
class CustomBattleOrder:
    def __init__(self, move):
        self.move = move

    @property
    def message(self):
        return f"/choose move {self.move.id}"

class QLearningPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.1, gamma=0.9, epsilon=0.2, **kwargs):
        super().__init__(battle_format=battle_format, max_concurrent_battles=1, **kwargs)
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # Persistence
    def save_model(self, filename="q_table.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.Q, f)
        except Exception as e:
            print(f"Save failed: {e}")

    def load_model(self, filename="q_table.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.Q = pickle.load(f)
            print(f"Loaded Q-table from {filename} with {len(self.Q)} states.")
        else:
            print("No saved model found. Starting from scratch.")

    # State Representation
    def get_state(self, battle):
        if battle.active_pokemon and battle.opponent_active_pokemon:
            my_hp = battle.active_pokemon.current_hp / battle.active_pokemon.max_hp
            opp_hp = battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp
        else:
            my_hp, opp_hp = 1.0, 1.0
        return (round(my_hp, 1), round(opp_hp, 1))

    # Move Selection
    def choose_move(self, battle):
        actions = battle.available_moves
        
        # Fallback if no moves available
        if not actions:
            return self.choose_random_move(battle)

        state = self.get_state(battle)
        selected_move = None

        # Epsilon-greedy policy
        if random.random() < self.epsilon or state not in self.Q:
            selected_move = random.choice(actions)
        else:
            q_values = self.Q[state]
            if q_values:
                # Basic logic to pick best move
                if len(q_values) == len(actions):
                    selected_move = actions[int(np.argmax(q_values))]
                else:
                    selected_move = random.choice(actions)
            else:
                selected_move = random.choice(actions)

        return CustomBattleOrder(selected_move)

    # Reward Function
    def _compute_reward(self, battle):
        return self.reward_computing_helper(battle, fainted_value=1, hp_value=0)

    # End of Battle Callback
    async def battle_end_callback(self, battle):
        pass

async def main():
    print("Creating Gen 1 players...")
    config = LocalhostServerConfiguration

    p1 = QLearningPlayer(battle_format="gen1randombattle", server_configuration=config)
    p1.load_model("q_table.pkl") 

    p2 = MaxBasePowerPlayer(battle_format="gen1randombattle",
                            server_configuration=config,
                            max_concurrent_battles=10)

    p1._start_challenging = True
    p2._start_challenging = False

    TOTAL_BATTLES = 1000
    print(f"Starting training loop for {TOTAL_BATTLES} battles...")

    for i in range(TOTAL_BATTLES):
        try:
            await p1.battle_against(p2, n_battles=1)
        except Exception as e:
            # If a battle crashes entirely, log it and move to the next one
            print(f"Battle {i+1} failed: {e}. Continuing...")
            continue

        if (i + 1) % 50 == 0:
            p1.save_model("q_table.pkl")
            print(f"Progress: {i+1}/{TOTAL_BATTLES} | Wins: {p1.n_won_battles}")

    print(f"Training finished. QAgent total wins: {p1.n_won_battles}")
    p1.save_model("q_table.pkl")


if __name__ == "__main__":
    asyncio.run(main())