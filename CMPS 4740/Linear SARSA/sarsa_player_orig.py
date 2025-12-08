import numpy as np
import pickle
import random
from poke_env.player.player import Player
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
# Import from the NEW features file
from features_orig import FeatureExtractor

# ==================================================================================
# 🚑 MONKEY PATCH
# ==================================================================================
_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request_moves):
    try:
        return _original_available_moves(self, request_moves)
    except AssertionError:
        return []
Pokemon.available_moves_from_request = patched_available_moves
# ==================================================================================

class LinearSARSAPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.01, gamma=0.99, tau=3.0, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        
        self.extractor = FeatureExtractor()
        # Small random weights
        self.weights = np.random.uniform(-0.01, 0.01, self.extractor.total_dim)
        
        self.alpha = alpha     
        self.gamma = gamma
        self.tau = tau         
        
        self._last_features = {} 
        self._last_q = {}
        
        # Memory
        self._last_opp_hp = {}
        self._last_my_hp = {}
        self._last_opp_status = {}

    def get_q(self, features):
        return np.dot(self.weights, features)

    def choose_move(self, battle):
        # 1. VALIDATION
        valid_actions = [] 
        if battle.active_pokemon and not battle.active_pokemon.fainted:
            for i, move in enumerate(battle.available_moves):
                if i < 4: valid_actions.append((move, i))
        
        for i, mon in enumerate(battle.available_switches):
             if i < 5: 
                 if not mon.fainted and not mon.active:
                     valid_actions.append((mon, 4 + i))

        if not valid_actions:
            return self.choose_random_move(battle)

        # 2. Q CALCULATION
        candidate_features = []
        q_values = []
        real_actions = []

        for action_obj, action_idx in valid_actions:
            phi = self.extractor.get_features(battle, action_obj)
            q = self.get_q(phi)
            candidate_features.append(phi)
            q_values.append(q)
            real_actions.append(action_obj)
            
        # 3. SELECTION (Softmax)
        if self.tau < 0.01:
            choice_idx = np.argmax(q_values)
        else:
            q_numpy = np.array(q_values)
            exp_q = np.exp((q_numpy - np.max(q_numpy)) / self.tau)
            probabilities = exp_q / np.sum(exp_q)
            choice_idx = np.random.choice(len(valid_actions), p=probabilities)
            
        chosen_action = real_actions[choice_idx]
        chosen_features = candidate_features[choice_idx]
        chosen_q = q_values[choice_idx]
        
        # 4. UPDATE
        battle_id = battle.battle_tag
        if battle_id in self._last_features:
            last_phi = self._last_features[battle_id]
            last_q = self._last_q[battle_id]
            
            reward = self.calculate_reward(battle)
            
            target = reward + self.gamma * chosen_q
            error = target - last_q
            self.weights += self.alpha * error * last_phi

        self._last_features[battle_id] = chosen_features
        self._last_q[battle_id] = chosen_q
        
        # SNAPSHOT
        if battle.opponent_active_pokemon:
            self._last_opp_hp[battle_id] = battle.opponent_active_pokemon.current_hp_fraction
            self._last_opp_status[battle_id] = battle.opponent_active_pokemon.status
        if battle.active_pokemon:
            self._last_my_hp[battle_id] = battle.active_pokemon.current_hp_fraction
        
        return self.create_order(chosen_action)

    def calculate_reward(self, battle):
        reward = 0
        battle_id = battle.battle_tag
        
        # 1. HP EFFICIENCY
        curr_opp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0
        curr_my = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0

        # Damage Dealt
        if battle_id in self._last_opp_hp:
            dmg = self._last_opp_hp[battle_id] - curr_opp
            if dmg > 0: reward += dmg * 20.0
            
        # Damage Taken
        if battle_id in self._last_my_hp:
            taken = self._last_my_hp[battle_id] - curr_my
            if taken > 0: reward -= taken * 20.0
        
        # 2. STATUS
        if battle.opponent_active_pokemon:
            curr_s = battle.opponent_active_pokemon.status
            prev_s = self._last_opp_status.get(battle_id, None)
            if prev_s is None and curr_s is not None:
                if curr_s.name in ['SLP', 'FRZ']: reward += 30.0 
                else: reward += 10.0

        # 3. FAINT
        opp_fainted = len([m for m in battle.opponent_team.values() if m.fainted])
        my_fainted = len([m for m in battle.team.values() if m.fainted])
        reward += (opp_fainted * 50.0) 
        reward -= (my_fainted * 50.0)

        return reward

    def battle_finished_callback(self, battle):
        battle_id = battle.battle_tag
        if battle_id in self._last_features:
            last_phi = self._last_features[battle_id]
            last_q = self._last_q[battle_id]
            
            reward = 1000 if battle.won else -1000
            
            target = reward
            error = target - last_q
            self.weights += self.alpha * error * last_phi
            
            del self._last_features[battle_id]
            del self._last_q[battle_id]
            if battle_id in self._last_opp_hp: del self._last_opp_hp[battle_id]
            if battle_id in self._last_my_hp: del self._last_my_hp[battle_id]
            if battle_id in self._last_opp_status: del self._last_opp_status[battle_id]

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                saved_weights = pickle.load(f)
                if hasattr(saved_weights, 'shape') and saved_weights.shape != self.weights.shape:
                    print(f"⚠️ SHAPE MISMATCH: Saved {saved_weights.shape} != Current {self.weights.shape}. Starting fresh.")
                else:
                    self.weights = saved_weights
                    print(f"Model loaded successfully from {path}")
        except FileNotFoundError:
            print("No model found, starting fresh.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting fresh.")