import random
import pickle
import os
import zlib
import numpy as np
from poke_env.player.player import Player
from poke_env.battle.pokemon import Pokemon
from features_v13 import DiscreteFeatureExtractor
import logging

# Fix Gen 1
_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request):
    try: return _original_available_moves(self, request)
    except AssertionError: return []
Pokemon.available_moves_from_request = patched_available_moves

class TabularQPlayerV13(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.1, gamma=0.99, lam=0.8, epsilon=0.1, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        
        self.extractor = DiscreteFeatureExtractor()
        
        # High-Level Q-Table: Dictionary {(state_key, action_hash): q_value}
        self.q_table = {}
        
        # Traces: {(state_key, action_hash): trace_value}
        self.active_traces = {}
        
        # Low-Level Agent (Switch Tactician)
        self.switch_table = {} 
        self.switch_traces = {}
        
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        
        # State Tracking
        self.last_state_key = None
        self.last_action_hash = None
        self.last_switch_context = None 
        self.last_switch_action_was_greedy = False
        
        # Dense Reward State
        self.last_reward_snapshot = None
        self.step_buffer = []

    # --- DENSE REWARD LOGIC ---
    def _get_dense_reward_snapshot(self, battle):
        my_hp = sum([mon.current_hp_fraction for mon in battle.team.values()])
        opp_hp = sum([mon.current_hp_fraction for mon in battle.opponent_team.values()])
        my_fainted = sum([1 for mon in battle.team.values() if mon.fainted])
        opp_fainted = sum([1 for mon in battle.opponent_team.values() if mon.fainted])
        return {
            'my_hp': my_hp, 'opp_hp': opp_hp, 
            'my_fainted': my_fainted, 'opp_fainted': opp_fainted
        }

    def _calculate_step_reward(self, current_snapshot):
        if self.last_reward_snapshot is None: return 0.0
        prev = self.last_reward_snapshot
        curr = current_snapshot
        reward = 0.0
        
        reward += (curr['opp_fainted'] - prev['opp_fainted']) * 0.1
        reward -= (curr['my_fainted'] - prev['my_fainted']) * 0.1
        
        opp_hp_lost = prev['opp_hp'] - curr['opp_hp']
        my_hp_lost = prev['my_hp'] - curr['my_hp']
        reward += 0.05 * (opp_hp_lost - my_hp_lost)
        
        return reward

    def pop_step_rewards(self):
        out = self.step_buffer
        self.step_buffer = []
        return out
    # --------------------------

    def get_q_value(self, state_key, action_hash):
        return self.q_table.get((state_key, action_hash), 0.0)

    def get_switch_value(self, opp_species, opp_hp, cand_species, cand_hp):
        key = (opp_species, opp_hp, cand_species, cand_hp)
        return self.switch_table.get(key, 0.0)

    def update_switch_value(self, context_key, reward, alpha_switch=0.1):
        old_val = self.switch_table.get(context_key, 0.0)
        new_val = old_val + alpha_switch * (reward - old_val)
        self.switch_table[context_key] = new_val

    def _update_traces_and_q(self, reward, max_next_q, next_action_is_greedy):
        # Watkins' Q(lambda) Update
        old_q = self.get_q_value(self.last_state_key, self.last_action_hash)
        delta = reward + self.gamma * max_next_q - old_q
        
        # 1. Update Switch Traces (TD Lambda)
        if self.last_switch_context:
            self.switch_traces[self.last_switch_context] = self.switch_traces.get(self.last_switch_context, 0.0) + 1.0

        switch_keys_to_remove = []
        for s_key, e_val in self.switch_traces.items():
            self.switch_table[s_key] = self.switch_table.get(s_key, 0.0) + self.alpha * delta * e_val
            
            if next_action_is_greedy and self.last_switch_action_was_greedy:
                new_e = e_val * self.gamma * self.lam
            else:
                new_e = 0.0
            
            if new_e < 0.001: switch_keys_to_remove.append(s_key)
            else: self.switch_traces[s_key] = new_e
        for k in switch_keys_to_remove: del self.switch_traces[k]

        # 2. Update Main Traces
        trace_key = (self.last_state_key, self.last_action_hash)
        self.active_traces[trace_key] = self.active_traces.get(trace_key, 0.0) + 1.0
        
        keys_to_remove = []
        for key, e_val in self.active_traces.items():
            self.q_table[key] = self.q_table.get(key, 0.0) + self.alpha * delta * e_val
            
            if next_action_is_greedy: new_e = e_val * self.gamma * self.lam
            else: new_e = 0.0
            
            if new_e < 0.001: keys_to_remove.append(key)
            else: self.active_traces[key] = new_e
            
        for k in keys_to_remove: del self.active_traces[k]

        if not next_action_is_greedy:
            self.active_traces.clear()
            self.switch_traces.clear()
            self.last_switch_action_was_greedy = False
            self.last_switch_context = None

    def choose_move(self, battle):
        # Calculate Step Reward
        current_snapshot = self._get_dense_reward_snapshot(battle)
        step_reward = self._calculate_step_reward(current_snapshot)
        if step_reward != 0:
            self.step_buffer.append(step_reward)
        self.last_reward_snapshot = current_snapshot

        # 1. State
        state_key = self.extractor.get_state_key(battle)
        
        # 2. Valid Actions
        possible_actions = []
        if not battle.force_switch and battle.active_pokemon and not battle.active_pokemon.fainted:
            for move in battle.available_moves:
                action_hash = zlib.adler32(move.id.encode())
                possible_actions.append((action_hash, move))
                
        if battle.available_switches:
            possible_actions.append((-1, None))
            
        if not possible_actions:
            return self.choose_random_move(battle)

        # 3. Get Q-Values
        q_values = [self.get_q_value(state_key, a_hash) for a_hash, _ in possible_actions]
        
        # 4. Selection
        if not q_values:
             max_q = 0.0
             greedy_idx = 0
        else:
             max_q = max(q_values)
             best_indices = [i for i, q in enumerate(q_values) if q == max_q]
             greedy_idx = random.choice(best_indices)

        if random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(possible_actions) - 1)
            is_greedy = (q_values[chosen_idx] == max_q)
        else:
            chosen_idx = greedy_idx
            is_greedy = True
            
        chosen_hash, chosen_move_obj = possible_actions[chosen_idx]

        # 5. Update Previous Step
        if self.last_state_key is not None:
            # IMPORTANT: We use the Step Reward for the update here
            self._update_traces_and_q(step_reward, max_q, is_greedy)

        self.last_state_key = state_key
        self.last_action_hash = chosen_hash
        
        if chosen_hash == -1:
            self.last_switch_context = None 
            return self._sub_agent_switch_learned(battle, is_greedy)
        else:
            self.last_switch_context = None 
            return self.create_order(chosen_move_obj)

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
            choice_hp = self.extractor.get_hp_bucket(choice.current_hp, choice.max_hp)
            is_sub_greedy = (choice.species == best_mon.species and choice_hp == self.extractor.get_hp_bucket(best_mon.current_hp, best_mon.max_hp)) if best_mon else False
        else:
            choice = best_mon if best_mon else random.choice(available)
            is_sub_greedy = True
            
        choice_hp = self.extractor.get_hp_bucket(choice.current_hp, choice.max_hp)
        self.last_switch_context = (opp_species, opp_hp, choice.species, choice_hp)
        self.last_switch_action_was_greedy = is_sub_greedy and parent_action_was_greedy
        return self.create_order(choice)

    def battle_finished_callback(self, battle):
        pass # We handle logic in choose_move or manual updates, not here.

    def _battle_finished(self, battle, won):
        # Calculate Final Step Reward
        current_snapshot = self._get_dense_reward_snapshot(battle)
        step_reward = self._calculate_step_reward(current_snapshot)
        
        # Win Reward
        win_reward = 1.0 if won else -1.0
        final_total_reward = step_reward + win_reward
        
        # Note: We do NOT append final_total_reward to step_buffer here.
        # The trainer loop adds the +1/-1 manually.
        
        if self.last_switch_context:
            self.update_switch_value(self.last_switch_context, final_total_reward, alpha_switch=0.1)

        if self.last_state_key is not None:
            self._update_traces_and_q(final_total_reward, 0.0, True)
            
        self.last_state_key = None
        self.last_action_hash = None
        self.active_traces.clear()
        self.switch_traces.clear()
        self.last_switch_context = None
        self.last_switch_action_was_greedy = False
        self.last_reward_snapshot = None
        
        # Win counters
        self._n_finished_battles += 1
        if won: self._n_won_battles += 1

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
            logging.critical(f"Loaded V13 Tables: Q ({len(self.q_table)}) Switch ({len(self.switch_table)})")
        except:
            logging.critical("Starting fresh V13 Tables.")