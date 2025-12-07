import random
import pickle
import os
import numpy as np
from poke_env.player.player import Player
from poke_env.battle.pokemon import Pokemon
from features_v10 import DiscreteFeatureExtractor
import logging

# Fix Gen 1
_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request):
    try: return _original_available_moves(self, request)
    except AssertionError: return []
Pokemon.available_moves_from_request = patched_available_moves

class TabularQPlayerV10(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.1, gamma=0.99, lam=0.9, epsilon=0.1, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        
        self.extractor = DiscreteFeatureExtractor()
        
        # --- High-Level Agent (Main Decision) ---
        self.q_table = {}
        self.active_traces = {}
        
        # --- Low-Level Agent (Switch Tactician) ---
        self.switch_table = {} 
        self.switch_traces = {}
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        
        # State Tracking (High Level)
        self.last_state_key = None
        self.last_action_idx = None
        
        # State Tracking (Low Level)
        self.last_switch_context = None 
        self.last_switch_action_was_greedy = False
        
        # Reward Tracking (New)
        self.prev_my_alive_count = 6
        self.prev_opp_alive_count = 6


    # --- Q-Value Helpers (Unchanged) ---
    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * 5 
        return self.q_table[state_key]

    def get_switch_value(self, opp_species, opp_hp, cand_species, cand_hp):
        key = (opp_species, opp_hp, cand_species, cand_hp)
        return self.switch_table.get(key, 0.0)

    def update_switch_value(self, context_key, reward, alpha_switch=0.1):
        # Monte Carlo Update for terminal switch reward
        old_val = self.switch_table.get(context_key, 0.0)
        new_val = old_val + alpha_switch * (reward - old_val)
        self.switch_table[context_key] = new_val

    def _calculate_faint_reward(self, battle):
        """Calculates intermediate reward based on faint counts."""
        my_alive = len([m for m in battle.team.values() if not m.fainted])
        opp_alive = len([m for m in battle.opponent_team.values() if not m.fainted])
        
        reward = 0.0
        
        # Check if opponent fainted this turn
        if opp_alive < self.prev_opp_alive_count:
            reward += 0.1
        
        # Check if we fainted this turn
        if my_alive < self.prev_my_alive_count:
            reward -= 0.1
            
        # Update tracking variables
        self.prev_my_alive_count = my_alive
        self.prev_opp_alive_count = opp_alive
        
        return reward

    def _update_traces_and_q(self, reward, state_key, next_action_is_greedy):
        # Standard Watkins' Q(lambda) Update
        old_q = self.q_table[self.last_state_key][self.last_action_idx]
        
        if state_key is None: max_next_q = 0.0
        else: max_next_q = max(self.get_q_values(state_key))
            
        delta = reward + self.gamma * max_next_q - old_q
        
        # Update Low-Level Agent (Switch Traces & Q)
        if self.last_switch_context:
            self.switch_traces[self.last_switch_context] = self.switch_traces.get(self.last_switch_context, 0.0) + 1.0

        switch_keys_to_remove = []
        for s_key in list(self.switch_traces.keys()):
            e_val = self.switch_traces[s_key]
            
            self.switch_table[s_key] = self.switch_table.get(s_key, 0.0) + self.alpha * delta * e_val
            
            if next_action_is_greedy and self.last_switch_action_was_greedy:
                new_e = e_val * self.gamma * self.lam
            else:
                new_e = 0.0
            
            if new_e < 0.001: switch_keys_to_remove.append(s_key)
            else: self.switch_traces[s_key] = new_e
        for k in switch_keys_to_remove: del self.switch_traces[k]

        # Update High-Level Agent (Main Traces & Q)
        if self.last_state_key not in self.active_traces: self.active_traces[self.last_state_key] = {}
        self.active_traces[self.last_state_key][self.last_action_idx] = self.active_traces[self.last_state_key].get(self.last_action_idx, 0.0) + 1.0
        
        keys_to_remove = []
        for s_key, actions_map in self.active_traces.items():
            if s_key not in self.q_table: self.q_table[s_key] = [0.0] * 5
            actions_to_pop = []
            
            for a_idx, e_val in actions_map.items():
                self.q_table[s_key][a_idx] += self.alpha * delta * e_val
                
                if next_action_is_greedy: new_e = e_val * self.gamma * self.lam
                else: new_e = 0.0
                
                if new_e < 0.001: actions_to_pop.append(a_idx)
                else: actions_map[a_idx] = new_e
            
            for a in actions_to_pop: del actions_map[a]
            if not actions_map: keys_to_remove.append(s_key)
                
        for k in keys_to_remove: del self.active_traces[k]
        
        if not next_action_is_greedy:
            self.active_traces.clear()
            self.switch_traces.clear()
            self.last_switch_action_was_greedy = False
            self.last_switch_context = None

    def choose_move(self, battle):
        # 1. Calculate and Consume Faint Reward
        reward = self._calculate_faint_reward(battle)
        
        # 2. State
        state_key = self.extractor.get_state_key(battle)
        q_values = self.get_q_values(state_key)
        legal_mask = self._get_legal_mask(battle)
        
        masked_q = [q if m == 1 else -float('inf') for q, m in zip(q_values, legal_mask)]
        greedy_action_idx = int(np.argmax(masked_q))
        
        # 3. Exploration/Greedy Choice
        if random.random() < self.epsilon:
            legal_indices = [i for i, x in enumerate(legal_mask) if x == 1]
            if not legal_indices: action_idx = 0
            else: action_idx = random.choice(legal_indices)
            is_greedy = (action_idx == greedy_action_idx)
        else:
            action_idx = greedy_action_idx
            is_greedy = True

        # 4. TD Update
        if self.last_state_key is not None:
            self._update_traces_and_q(reward, state_key, is_greedy)

        self.last_state_key = state_key
        self.last_action_idx = action_idx
        
        # 5. Execute Action
        if action_idx < 4:
            self.last_switch_context = None 
            
            active_mon = battle.active_pokemon
            if not active_mon: return self.choose_random_move(battle)
            all_known_moves = sorted(list(active_mon.moves.keys()))
            
            if action_idx < len(all_known_moves):
                target_move_id = all_known_moves[action_idx]
                for move in battle.available_moves:
                    if move.id == target_move_id:
                        return self.create_order(move)
                return self.choose_random_move(battle)
            else:
                return self.choose_random_move(battle)
        else:
            return self._sub_agent_switch_learned(battle, is_greedy)

    def _get_legal_mask(self, battle):
        # Logic remains the same
        mask = [0] * 5
        active_mon = battle.active_pokemon
        if not battle.force_switch and active_mon and not active_mon.fainted:
            all_known_moves = sorted(list(active_mon.moves.keys()))
            available_ids = {m.id for m in battle.available_moves}
            for i in range(4):
                if i < len(all_known_moves):
                    if all_known_moves[i] in available_ids:
                        mask[i] = 1
        if battle.available_switches:
            mask[4] = 1 
        if sum(mask) == 0: return [1, 0, 0, 0, 0]
        return mask

    def _sub_agent_switch_learned(self, battle, parent_action_was_greedy):
        available = battle.available_switches
        if not available: return self.choose_random_move(battle)
        
        opp_mon = battle.opponent_active_pokemon
        opp_species = opp_mon.species if opp_mon else "None"
        opp_hp = self.extractor.get_hp_bucket(opp_mon.current_hp, opp_mon.max_hp) if opp_mon else 0
        
        best_val = -float('inf')
        best_mon = None
        
        for mon in available:
            cand_hp = self.extractor.get_hp_bucket(mon.current_hp, mon.max_hp)
            val = self.get_switch_value(opp_species, opp_hp, mon.species, cand_hp)
            
            if val > best_val:
                best_val = val
                best_mon = mon
        
        if random.random() < self.epsilon:
            choice = random.choice(available)
            is_sub_greedy = (choice.species == best_mon.species if best_mon else False)
        else:
            choice = best_mon if best_mon else random.choice(available)
            is_sub_greedy = True
            
        choice_hp = self.extractor.get_hp_bucket(choice.current_hp, choice.max_hp)
        self.last_switch_context = (opp_species, opp_hp, choice.species, choice_hp)
        self.last_switch_action_was_greedy = is_sub_greedy and parent_action_was_greedy
        return self.create_order(choice)

    def battle_finished_callback(self, battle):
        # Final Reward: Sparse Win/Loss (+1/-1)
        # Note: This terminal reward accumulates on top of any faint rewards already collected.
        reward = 1.0 if battle.won else -1.0
        
        # 1. Update Low-Level Agent (Terminal MC Update)
        if self.last_switch_context:
            self.update_switch_value(self.last_switch_context, reward, alpha_switch=0.1)

        # 2. Update High-Level Agent (Terminal TD Update)
        if self.last_state_key is not None:
            self._update_traces_and_q(reward, None, True)
            
        # Reset tracking for new battle
        self.prev_my_alive_count = 6
        self.prev_opp_alive_count = 6
        
        self.last_state_key = None
        self.last_action_idx = None
        self.active_traces.clear()
        self.switch_traces.clear()
        self.last_switch_context = None
        self.last_switch_action_was_greedy = False

    def save_table(self, path):
        data = {'q': self.q_table, 'switch': self.switch_table}
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_table(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q']
                self.switch_table = data['switch']
            logging.critical(f"Loaded Q-Table ({len(self.q_table)} states) and Switch-Table ({len(self.switch_table)} pairs).")
        except:
            logging.critical("Starting fresh Tables.")