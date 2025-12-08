import asyncio
import numpy as np
import random
import pickle
import os
import uuid
import matplotlib.pyplot as plt 

from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.battle import pokemon 
from poke_env.ps_client.account_configuration import AccountConfiguration

# Patch to fix Gen 1 crashes
_original_available_moves = pokemon.Pokemon.available_moves_from_request

def patched_available_moves(self, request_moves):
    try:
        return _original_available_moves(self, request_moves)
    except AssertionError:
        return []

pokemon.Pokemon.available_moves_from_request = patched_available_moves

class CustomBattleOrder:
    def __init__(self, move):
        self.move = move
    @property
    def message(self):
        return f"/choose move {self.move.id}"


class QLearningPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.0, gamma=0.9, epsilon=0.0, **kwargs):
        super().__init__(battle_format=battle_format, max_concurrent_battles=1, **kwargs)
        self.Q = {}
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        self.previous_state = None
        self.previous_action = None

    # Load from the NO SWITCH table
    def load_model(self, filename="q_table_no_switch.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.Q = pickle.load(f)
                print(f"Loaded Q-table with {len(self.Q)} states.")
            except (EOFError, pickle.UnpicklingError):
                print("Error: Saved model was corrupted.")
                self.Q = {}
        else:
            print(f"Error: {filename} not found! You must run the old training script first.")

    # State encoding
    def get_state(self, battle):
        def get_hp_bucket(pokemon):
            if not pokemon or pokemon.current_hp == 0: return 0
            return int(pokemon.current_hp / pokemon.max_hp * 4) + 1

        def get_type(pokemon):
            if not pokemon: return "none"
            return str(pokemon.type_1).split(".")[-1].lower()

        def get_status_bucket(pokemon):
            if not pokemon or not pokemon.status: return 0
            s = str(pokemon.status).split('.')[-1].lower()
            if s in ['slp', 'frz']: return 2 
            if s in ['par', 'brn', 'psn', 'tox']: return 1 
            return 0

        def am_i_faster(active, opp_active):
            if not active or not opp_active: return 0
            return 1 if active.base_stats['spe'] > opp_active.base_stats['spe'] else 0

        my_hp = get_hp_bucket(battle.active_pokemon)
        opp_hp = get_hp_bucket(battle.opponent_active_pokemon)
        
        my_type = get_type(battle.active_pokemon)
        opp_type = get_type(battle.opponent_active_pokemon)
        
        opp_status = get_status_bucket(battle.opponent_active_pokemon)
        im_faster = am_i_faster(battle.active_pokemon, battle.opponent_active_pokemon)

        return (my_hp, opp_hp, my_type, opp_type, opp_status, im_faster)

    # Move selection logic
    def choose_move(self, battle):
        current_state = self.get_state(battle)
        
        # Action masking
        available_moves = battle.available_moves
        good_moves = []

        if available_moves:
            for move in available_moves:
                if move.id in ['recover', 'softboiled', 'rest', 'milkdrink'] and \
                   battle.active_pokemon.current_hp == battle.active_pokemon.max_hp:
                    continue
                
                if move.category.name == 'STATUS' and battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
                     if move.target == 'normal' or move.target == 'any': 
                         continue
                
                good_moves.append(move)
        
        actions = good_moves if good_moves else available_moves
        if not actions:
            return self.choose_random_move(battle)

        # Pure greedy selection
        selected_move = None
        
        if current_state in self.Q:
            q_values = self.Q[current_state]
            if q_values:
                best_move_id = max(q_values, key=q_values.get)
                matching_moves = [m for m in actions if m.id == best_move_id]
                selected_move = matching_moves[0] if matching_moves else random.choice(actions)
            else:
                selected_move = random.choice(actions)
        else:
            selected_move = random.choice(actions)

        return CustomBattleOrder(selected_move)


async def main():
    print("Starting Testing Mode (No Switching)...")
    config = LocalhostServerConfiguration

    p1_name = f"TestBot_{str(uuid.uuid4())[:8]}"
    p2_name = f"MaxBot_{str(uuid.uuid4())[:8]}"

    p1 = QLearningPlayer(
        battle_format="gen1randombattle", 
        server_configuration=config,
        account_configuration=AccountConfiguration(p1_name, None),
        epsilon=0.0, 
        alpha=0.0    
    )
    
    # Load the NO SWITCH brain
    p1.load_model("q_table_no_switch.pkl") 

    p2 = MaxBasePowerPlayer(
        battle_format="gen1randombattle",
        server_configuration=config,
        account_configuration=AccountConfiguration(p2_name, None),
        max_concurrent_battles=1
    )

    p1._start_challenging = True
    p2._start_challenging = False

    # Test configuration
    TOTAL_BATTLES = 1000  
    BATCH_SIZE = 50       
    
    x_axis = []  
    y_rolling = []
    prev_win_count = p1.n_won_battles

    print(f"Running {TOTAL_BATTLES} test battles against MaxBasePowerPlayer...")

    for i in range(TOTAL_BATTLES):
        try:
            await p1.battle_against(p2, n_battles=1)
        except Exception as e:
            print(f"Battle {i+1} failed. Continuing...")
            continue

        if (i + 1) % BATCH_SIZE == 0:
            current_win_count = p1.n_won_battles
            wins_in_batch = current_win_count - prev_win_count
            win_rate_batch = wins_in_batch / BATCH_SIZE
            
            x_axis.append(i + 1)
            y_rolling.append(win_rate_batch)
            
            prev_win_count = current_win_count
            
            print(f"Test Batch {i+1}: Rolling Win Rate {win_rate_batch:.2%}")

    print("Testing Finished.")
    print(f"Final Win Rate: {p1.n_won_battles / TOTAL_BATTLES:.2%}")
    
    # Plotting
    try:
        print("Generating test results graph...")
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, y_rolling, label=f'Rolling Win Rate (Last {BATCH_SIZE})', color='#1f77b4', marker='o', linestyle='-')
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% Break-even')
        plt.title(f'Agent Performance (No Switching Test)')
        plt.xlabel('Number of Battles')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1.0) 
        plt.grid(True)
        plt.legend()
        
        plt.savefig("test_results_no_switch.png")
        print("Graph saved as 'test_results_no_switch.png'. Check your folder!")
        
    except Exception as e:
        print(f"Could not plot graph: {e}")


if __name__ == "__main__":
    asyncio.run(main())