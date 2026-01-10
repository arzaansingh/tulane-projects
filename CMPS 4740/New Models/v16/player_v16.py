import random
import pickle
import os
import zlib
import numpy as np
import logging
import gc
import math
from poke_env.player.player import Player
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move_category import MoveCategory
from features_v16 import AdvancedFeatureExtractor

# Fix Gen 1/4 moves issue
_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request):
    try: return _original_available_moves(self, request)
    except AssertionError: return []
Pokemon.available_moves_from_request = patched_available_moves

class HeuristicEngine:
    """
    Encapsulates the logic from SimpleHeuristicsPlayer to estimate move/switch quality.
    """
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    
    @staticmethod
    def _stat_estimation(mon, stat):
        if stat not in mon.boosts: return mon.base_stats.get(stat, 0)
        boost = mon.boosts[stat]
        if boost > 1: multiplier = (2 + boost) / 2
        else: multiplier = 2 / (2 - boost)
        return ((2 * mon.base_stats.get(stat, 100) + 31) + 5) * multiplier

    @staticmethod
    def _estimate_matchup(mon, opponent):
        if not opponent: return 0
        # Type effectiveness
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max([mon.damage_multiplier(t) for t in opponent.types if t is not None])
        
        # Speed
        if mon.base_stats.get("spe", 0) > opponent.base_stats.get("spe", 0):
            score += HeuristicEngine.SPEED_TIER_COEFICIENT
        elif opponent.base_stats.get("spe", 0) > mon.base_stats.get("spe", 0):
            score -= HeuristicEngine.SPEED_TIER_COEFICIENT
            
        # HP
        score += mon.current_hp_fraction * HeuristicEngine.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * HeuristicEngine.HP_FRACTION_COEFICIENT
        return score

    @staticmethod
    def get_move_score(battle, move, active, opponent):
        if not opponent or not active: return 0.0
        
        # Determine stats for damage calculation
        if move.category == MoveCategory.PHYSICAL:
            atk = HeuristicEngine._stat_estimation(active, "atk")
            defn = HeuristicEngine._stat_estimation(opponent, "def")
        else:
            atk = HeuristicEngine._stat_estimation(active, "spa")
            defn = HeuristicEngine._stat_estimation(opponent, "spd")
            
        ratio = atk / defn if defn > 0 else 1.0
        
        # Components
        stab = 1.5 if move.type in active.types else 1.0
        type_eff = opponent.damage_multiplier(move.type)
        base_power = move.base_power
        
        # Heuristic Score Formula
        score = base_power * stab * ratio * move.accuracy * move.expected_hits * type_eff
        return score

    @staticmethod
    def get_switch_score(battle, candidate, opponent):
        return HeuristicEngine._estimate_matchup(candidate, opponent)

class TabularQPlayerV16(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.1, gamma=0.99, lam=0.8, epsilon=0.1, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        
        self.extractor = AdvancedFeatureExtractor()
        
        # Tables
        self.q_table = {}
        self.switch_table = {} 
        
        self.active_traces = {}
        self.switch_traces = {}
        
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        
        self.last_state_key = None
        self.last_action_hash = None
        self.last_switch_context = None 
        self.last_switch_action_was_greedy = False
        
        self.last_reward_snapshot = None
        self.step_buffer = []

    # --- INITIALIZATION LOGIC ---
    def _initialize_state_if_needed(self, battle, state_key, possible_actions):
        """
        If any action for this state is missing, calculate Heuristic Scores, 
        Softmax them, and initialize Q-values.
        """
        missing_hashes = [h for h, _ in possible_actions if (state_key, h) not in self.q_table]
        
        if not missing_hashes:
            return # State already fully explored

        # 1. Calculate Raw Scores for ALL actions (to normalize properly)
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        raw_scores = []
        for action_hash, move_obj in possible_actions:
            if action_hash == -1: # Switch Action
                # Give switching a baseline score (e.g., average of available switches)
                # Or a small penalty/bonus depending on if we are trapped
                score = 50.0 # Arbitrary mid-range score for "Switching general option"
            else:
                score = HeuristicEngine.get_move_score(battle, move_obj, active, opponent)
            raw_scores.append(score)
            
        # 2. Softmax Normalization
        # Subtract max for numerical stability
        max_s = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp((s - max_s) / 10.0) for s in raw_scores] # Div by 10 to flatten extreme differences
        sum_exp = sum(exp_scores)
        
        normalized_scores = [e / sum_exp for e in exp_scores]
        
        # 3. Store ONLY the missing ones (or overwrite? "Initialize" implies fill empty)
        for i, (action_hash, _) in enumerate(possible_actions):
            if (state_key, action_hash) not in self.q_table:
                # Initialize with the heuristic probability [0.0, 1.0]
                self.q_table[(state_key, action_hash)] = normalized_scores[i]

    def _initialize_switch_if_needed(self, battle, candidates):
        """
        Initialize Switch Sub-Agent Q-values using matchup heuristics.
        """
        opponent = battle.opponent_active_pokemon
        
        # Check which candidates are missing from table
        missing = []
        contexts = []
        for mon in candidates:
            ctx = self.extractor.get_sub_state(battle, mon)
            contexts.append(ctx)
            if ctx not in self.switch_table:
                missing.append((mon, ctx))
                
        if not missing: return

        # Calculate scores
        raw_scores = []
        for mon in candidates:
            score = HeuristicEngine.get_switch_score(battle, mon, opponent)
            raw_scores.append(score)
            
        # Softmax
        max_s = max(raw_scores) if raw_scores else 0
        exp_scores = [math.exp(s - max_s) for s in raw_scores]
        sum_exp = sum(exp_scores)
        normalized_scores = [e / sum_exp for e in exp_scores]
        
        # Store
        for i, ctx in enumerate(contexts):
            if ctx not in self.switch_table:
                self.switch_table[ctx] = normalized_scores[i]

    # --- STANDARD Q-LEARNING METHODS ---
    def _get_dense_reward_snapshot(self, battle):
        my_hp = sum([mon.current_hp_fraction for mon in battle.team.values()])
        opp_hp = sum([mon.current_hp_fraction for mon in battle.opponent_team.values()])
        my_fainted = sum([1 for mon in battle.team.values() if mon.fainted])
        opp_fainted = sum([1 for mon in battle.opponent_team.values() if mon.fainted])
        my_status = sum([1 for mon in battle.team.values() if mon.status])
        opp_status = sum([1 for mon in battle.opponent_team.values() if mon.status])
        my_boosts = 0
        if battle.active_pokemon:
            my_boosts = sum(battle.active_pokemon.boosts.values())
        return {'my_hp': my_hp, 'opp_hp': opp_hp, 'my_fainted': my_fainted, 
                'opp_fainted': opp_fainted, 'my_status': my_status, 
                'opp_status': opp_status, 'my_boosts': my_boosts}

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
        new_opp_status = curr['opp_status'] - prev['opp_status']
        new_my_status = curr['my_status'] - prev['my_status']
        reward += 0.01 * (new_opp_status - new_my_status)
        boost_change = curr['my_boosts'] - prev['my_boosts']
        reward += 0.01 * boost_change
        return reward

    def pop_step_rewards(self):
        out = self.step_buffer
        self.step_buffer = []
        return out

    def get_q_value(self, state_key, action_hash):
        # NOTE: Initialization happens in choose_move before this is called
        return self.q_table.get((state_key, action_hash), 0.0)

    def get_switch_value(self, sub_state_key):
        return self.switch_table.get(sub_state_key, 0.0)

    def update_switch_value(self, context_key, reward, alpha_switch=0.1):
        old_val = self.switch_table.get(context_key, 0.0)
        new_val = old_val + alpha_switch * (reward - old_val)
        self.switch_table[context_key] = new_val

    def _update_traces_and_q(self, reward, max_next_q, next_action_is_greedy):
        old_q = self.get_q_value(self.last_state_key, self.last_action_hash)
        delta = reward + self.gamma * max_next_q - old_q
        
        if self.last_switch_context:
            self.switch_traces[self.last_switch_context] = self.switch_traces.get(self.last_switch_context, 0.0) + 1.0

        switch_keys_to_remove = []
        for s_key, e_val in self.switch_traces.items():
            self.switch_table[s_key] = self.switch_table.get(s_key, 0.0) + self.alpha * delta * e_val
            new_e = e_val * self.gamma * self.lam if (next_action_is_greedy and self.last_switch_action_was_greedy) else 0.0
            if new_e < 0.001: switch_keys_to_remove.append(s_key)
            else: self.switch_traces[s_key] = new_e
        for k in switch_keys_to_remove: del self.switch_traces[k]

        trace_key = (self.last_state_key, self.last_action_hash)
        self.active_traces[trace_key] = self.active_traces.get(trace_key, 0.0) + 1.0
        
        keys_to_remove = []
        for key, e_val in self.active_traces.items():
            self.q_table[key] = self.q_table.get(key, 0.0) + self.alpha * delta * e_val
            new_e = e_val * self.gamma * self.lam if next_action_is_greedy else 0.0
            if new_e < 0.001: keys_to_remove.append(key)
            else: self.active_traces[key] = new_e
        for k in keys_to_remove: del self.active_traces[k]

        if not next_action_is_greedy:
            self.active_traces.clear()
            self.switch_traces.clear()
            self.last_switch_action_was_greedy = False
            self.last_switch_context = None

    def choose_move(self, battle):
        current_snapshot = self._get_dense_reward_snapshot(battle)
        step_reward = self._calculate_step_reward(current_snapshot)
        if step_reward != 0: self.step_buffer.append(step_reward)
        self.last_reward_snapshot = current_snapshot

        state_key = self.extractor.get_master_state(battle)
        
        possible_actions = []
        if not battle.force_switch and battle.active_pokemon and not battle.active_pokemon.fainted:
            for move in battle.available_moves:
                action_hash = zlib.adler32(move.id.encode())
                possible_actions.append((action_hash, move))
        if battle.available_switches:
            possible_actions.append((-1, None))
            
        if not possible_actions:
            return self.choose_random_move(battle)

        # --- V16 UPGRADE: INITIALIZE WITH HEURISTICS ---
        self._initialize_state_if_needed(battle, state_key, possible_actions)

        q_values = [self.get_q_value(state_key, a_hash) for a_hash, _ in possible_actions]
        
        if not q_values:
             max_q = 0.0; greedy_idx = 0
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

        if self.last_state_key is not None:
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
        
        # --- V16 UPGRADE: INITIALIZE SWITCHES ---
        self._initialize_switch_if_needed(battle, available)
        
        best_val = -float('inf')
        best_mon = None
        best_context = None
        
        for mon in available:
            sub_state_key = self.extractor.get_sub_state(battle, mon)
            val = self.get_switch_value(sub_state_key)
            if val > best_val:
                best_val = val
                best_mon = mon
                best_context = sub_state_key
        
        if random.random() < self.epsilon:
            choice = random.choice(available)
            choice_context = self.extractor.get_sub_state(battle, choice)
            is_sub_greedy = (choice.species == best_mon.species) if best_mon else False
        else:
            choice = best_mon if best_mon else random.choice(available)
            choice_context = best_context if best_context else self.extractor.get_sub_state(battle, choice)
            is_sub_greedy = True
            
        self.last_switch_context = choice_context
        self.last_switch_action_was_greedy = is_sub_greedy and parent_action_was_greedy
        return self.create_order(choice)

    def battle_finished_callback(self, battle):
        pass 

    def _battle_finished(self, battle, won):
        current_snapshot = self._get_dense_reward_snapshot(battle)
        step_reward = self._calculate_step_reward(current_snapshot)
        win_reward = 1.0 if won else -1.0
        final_total_reward = step_reward + win_reward
        
        if self.last_switch_context:
            self.update_switch_value(self.last_switch_context, final_total_reward, alpha_switch=0.1)

        if self.last_state_key is not None:
            self._update_traces_and_q(final_total_reward, 0.0, True)
            
        self.last_state_key = None; self.last_action_hash = None
        self.active_traces.clear(); self.switch_traces.clear()
        self.last_switch_context = None; self.last_switch_action_was_greedy = False
        self.last_reward_snapshot = None
        
        self._n_finished_battles += 1
        if won: self._n_won_battles += 1

    def save_table(self, path):
        gc.disable()
        try:
            data = {'q': self.q_table, 'switch': self.switch_table}
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            gc.enable()

    def load_table(self, path):
        gc.disable()
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data.get('q', {})
                self.switch_table = data.get('switch', {})
            logging.critical(f"Loaded V16 Tables: Q ({len(self.q_table)}) Switch ({len(self.switch_table)})")
        except Exception as e:
            logging.critical(f"Starting fresh V16. Error: {e}")
        finally:
            gc.enable()