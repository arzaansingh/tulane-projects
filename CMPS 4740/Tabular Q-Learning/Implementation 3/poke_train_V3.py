import asyncio
import numpy as np
import random
import pickle
import os
import uuid
import matplotlib.pyplot as plt 
import logging
from typing import Dict, Tuple, Any

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

# Suppress console clutter
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)

class CustomBattleOrder:
    def __init__(self, action):
        self.action = action
    
    @property
    def message(self):
        if hasattr(self.action, "species"):
            return f"/choose switch {self.action.species}"
        return f"/choose move {self.action.id}"


class QLearningPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.1, gamma=0.9, epsilon=0.5, **kwargs):
        super().__init__(battle_format=battle_format, max_concurrent_battles=1, **kwargs)
        self.Q = {}
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        self.previous_state = None
        self.previous_action_key = None 

        # Reward Tracking
        self.last_my_hp = 1.0
        self.last_opp_hp = 1.0
        self.last_opp_fainted = 0
        self.last_my_fainted = 0
        self.last_opp_status = None
        
        # Repetition Tracking
        self.move_history_counter = 0
        self.last_move_id = None

    # Saves the Q-table to a file
    def save_model(self, filename="q_table_switch.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.Q.copy(), f)
        except Exception as e:
            print(f"Save failed (ignoring): {e}")

    # Loads the Q-table from a file if it exists
    def load_model(self, filename="q_table_switch.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.Q = pickle.load(f)
                print(f"Loaded Q-table with {len(self.Q)} states.")
            except (EOFError, pickle.UnpicklingError):
                print("Saved model was corrupted/empty. Starting from scratch.")
                self.Q = {}
        else:
            print(f"No {filename} found. Starting from scratch.")

    # Helper to get a string key for an action
    def get_action_key(self, action):
        if hasattr(action, "species"):
            return f"switch_{action.species.lower()}" 
        return action.id 

    # encode the battle state into a tuple
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

    # Calculates the reward based on the battle outcome and turn events
    def _compute_reward(self, battle):
        reward = 0.0
        
        if battle.won: return 100.0
        if battle.lost: return -100.0

        def get_hp_percent(pokemon):
            if not pokemon or pokemon.max_hp == 0: return 0.0
            return pokemon.current_hp / pokemon.max_hp

        # Calculate HP changes
        my_current_hp = get_hp_percent(battle.active_pokemon)
        opp_current_hp = get_hp_percent(battle.opponent_active_pokemon)

        damage_dealt = self.last_opp_hp - opp_current_hp
        damage_taken = self.last_my_hp - my_current_hp
        reward += (damage_dealt - damage_taken) * 50.0 

        # Calculate Fainted counts
        opp_fainted_count = len([mon for mon in battle.opponent_team.values() if mon.fainted])
        my_fainted_count = len([mon for mon in battle.team.values() if mon.fainted])

        if opp_fainted_count > self.last_opp_fainted: reward += 20.0
        if my_fainted_count > self.last_my_fainted: reward -= 20.0

        # Check for status changes
        opp_status = battle.opponent_active_pokemon.status if battle.opponent_active_pokemon else None
        if opp_status != self.last_opp_status and opp_status is not None:
            reward += 10.0 
            
        # Penalty for using the same move too many times
        if self.move_history_counter >= 3:
            reward -= 30.0 

        return reward

    # Main logic to choose a move
    def choose_move(self, battle):
        current_state = self.get_state(battle)
        reward = self._compute_reward(battle)
        
        def get_hp_percent(pokemon):
            if not pokemon or pokemon.max_hp == 0: return 0.0
            return pokemon.current_hp / pokemon.max_hp
            
        self.last_my_hp = get_hp_percent(battle.active_pokemon)
        self.last_opp_hp = get_hp_percent(battle.opponent_active_pokemon)
        self.last_opp_fainted = len([mon for mon in battle.opponent_team.values() if mon.fainted])
        self.last_my_fainted = len([mon for mon in battle.team.values() if mon.fainted])
        self.last_opp_status = battle.opponent_active_pokemon.status if battle.opponent_active_pokemon else None

        # Bellman equation update
        if self.previous_state and self.previous_action_key:
            if self.previous_state not in self.Q: self.Q[self.previous_state] = {}
            if self.previous_action_key not in self.Q[self.previous_state]: 
                self.Q[self.previous_state][self.previous_action_key] = 0.0
            
            max_q_current = 0.0
            if current_state in self.Q and self.Q[current_state]:
                max_q_current = max(self.Q[current_state].values())

            old_q = self.Q[self.previous_state][self.previous_action_key]
            new_q = old_q + self.alpha * (reward + self.gamma * max_q_current - old_q)
            self.Q[self.previous_state][self.previous_action_key] = new_q

        available_moves = battle.available_moves
        available_switches = battle.available_switches
        possible_actions = []
        
        # Filter moves
        if available_moves:
            for move in available_moves:
                if move.id in ['recover', 'softboiled', 'rest', 'milkdrink'] and \
                   battle.active_pokemon.current_hp == battle.active_pokemon.max_hp: continue
                
                if move.category.name == 'STATUS' and battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
                     if move.target == 'normal' or move.target == 'any': continue
                possible_actions.append(move)
        
        # Filter switches
        if available_switches:
            for pokemon in available_switches:
                if not pokemon.fainted and not pokemon.active:
                    possible_actions.append(pokemon)

        if not possible_actions:
            return self.choose_random_move(battle)

        selected_action = None
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon or current_state not in self.Q:
            selected_action = random.choice(possible_actions)
        else:
            q_values = self.Q[current_state]
            if q_values:
                best_action = None
                best_val = -float('inf')
                for action in possible_actions:
                    key = self.get_action_key(action)
                    val = q_values.get(key, 0.0) 
                    if val > best_val:
                        best_val = val
                        best_action = action
                selected_action = best_action if best_action else random.choice(possible_actions)
            else:
                selected_action = random.choice(possible_actions)

        self.previous_state = current_state
        self.previous_action_key = self.get_action_key(selected_action)
        
        # Update repetition counter
        if self.previous_action_key == self.last_move_id:
            self.move_history_counter += 1
        else:
            self.move_history_counter = 1
            self.last_move_id = self.previous_action_key
        
        return CustomBattleOrder(selected_action)
    
    # Reset state variables at end of battle
    async def battle_end_callback(self, battle):
        self.previous_state = None
        self.previous_action_key = None
        self.last_my_hp = 1.0
        self.last_opp_hp = 1.0
        self.last_opp_fainted = 0
        self.last_my_fainted = 0
        self.last_opp_status = None
        
        self.move_history_counter = 0
        self.last_move_id = None


async def main():
    print("Creating Gen 1 players...")
    config = LocalhostServerConfiguration

    p1_name = f"QBot_{str(uuid.uuid4())[:8]}"
    p2_name = f"MaxBot_{str(uuid.uuid4())[:8]}"

    # Training configuration
    START_EPSILON = 1
    MIN_EPSILON = 0.05
    TOTAL_BATTLES = 100 
    BATCH_SIZE = 100
    EXPLORATION_PHASE = 0.8 

    p1 = QLearningPlayer(
        battle_format="gen1randombattle", 
        server_configuration=config,
        account_configuration=AccountConfiguration(p1_name, None),
        epsilon=START_EPSILON 
    )
    
    p1.load_model("q_table_switch_updated.pkl") 

    p2 = MaxBasePowerPlayer(
        battle_format="gen1randombattle",
        server_configuration=config,
        account_configuration=AccountConfiguration(p2_name, None),
        max_concurrent_battles=1
    )

    p1._start_challenging = True
    p2._start_challenging = False
    
    # Calculate decay rate
    battles_to_decay = TOTAL_BATTLES * EXPLORATION_PHASE
    epsilon_decay_amount = (START_EPSILON - MIN_EPSILON) / battles_to_decay
    
    x_axis = []  
    y_rolling = []
    prev_win_count = p1.n_won_battles

    print(f"Starting training loop for {TOTAL_BATTLES} battles...")
    print(f"Starting Epsilon: {START_EPSILON} (Decaying to {MIN_EPSILON})")
    
    for i in range(TOTAL_BATTLES):
        try:
            await p1.battle_against(p2, n_battles=1)
            
            if p1.epsilon > MIN_EPSILON:
                p1.epsilon -= epsilon_decay_amount
            else:
                p1.epsilon = MIN_EPSILON

        except Exception as e:
            print(f"Battle {i+1} failed. Continuing...")
            continue

        if (i + 1) % BATCH_SIZE == 0:
            p1.save_model("q_table_switch.pkl")
            
            current_win_count = p1.n_won_battles
            wins_in_batch = current_win_count - prev_win_count
            win_rate_batch = wins_in_batch / BATCH_SIZE
            
            x_axis.append(i + 1)
            y_rolling.append(win_rate_batch)
            prev_win_count = current_win_count
            
            print(f"Batch {i+1}: Rolling {win_rate_batch:.1%} | Eps: {p1.epsilon:.4f}")

    print("Training finished.")
    print(f"Final Total Wins: {p1.n_won_battles}")
    p1.save_model("q_table_switch.pkl")
    
    try:
        print("Generating learning curve...")
        plt.figure(figsize=(12, 7))
        plt.plot(x_axis, y_rolling, label=f'Rolling Win Rate (Last {BATCH_SIZE})', color='#1f77b4', alpha=0.8, linewidth=2.0)
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% Break-even', alpha=0.5)
        plt.title(f'Q-Learning Training Progress (Switch Enabled)')
        plt.xlabel('Number of Battles')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1.0) 
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.savefig("learning_curve_switch.png")
        print("Graph saved as 'learning_curve_switch.png'. Check your folder!")
    except Exception as e:
        print(f"Could not plot graph: {e}")

if __name__ == "__main__":
    asyncio.run(main())