import numpy as np
import random
import torch
import torch.nn as nn  # <--- Added missing import
from poke_env.player.player import Player
from poke_env.battle.pokemon import Pokemon
from features_v4 import FeatureExtractor
from dqn_model import DQN, ReplayBuffer

# Fix Gen 1
_original_available_moves = Pokemon.available_moves_from_request
def patched_available_moves(self, request):
    try:
        return _original_available_moves(self, request)
    except AssertionError:
        return []
Pokemon.available_moves_from_request = patched_available_moves

class DQNPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", epsilon=1.0, **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        
        self.extractor = FeatureExtractor()
        
        # Setup Device (MPS for Mac M-series, else CPU)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Neural Network
        self.model = DQN(self.extractor.total_dim).to(self.device)
        self.target_model = DQN(self.extractor.total_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        
        self.memory = ReplayBuffer(capacity=50000)
        self.batch_size = 512
        self.gamma = 0.999
        self.epsilon = epsilon
        
        self._last_features = {} # phi(s, a)
        
        # Reporting
        self.last_battle_won = False

    def choose_move(self, battle):
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

        # 1. Calculate Q-values for all possible actions
        candidate_features = []
        for action_obj, _ in valid_actions:
            phi = self.extractor.get_features(battle, action_obj)
            candidate_features.append(phi)
        
        # Convert to Tensor for batch prediction
        features_tensor = torch.FloatTensor(np.array(candidate_features)).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(features_tensor).cpu().numpy().flatten()
            
        # 2. Epsilon-Greedy Selection
        if random.random() < self.epsilon:
            choice_idx = random.randint(0, len(valid_actions) - 1)
        else:
            choice_idx = np.argmax(q_values)
            
        chosen_action = valid_actions[choice_idx][0] # Action Object
        chosen_phi = candidate_features[choice_idx]
        
        # 3. Store Transition
        battle_id = battle.battle_tag
        if battle_id in self._last_features:
            last_phi = self._last_features[battle_id]
            # Reward is 0 for intermediate steps
            # Note: We pass '0' for action because it's embedded in the state features
            self.memory.push(last_phi, 0, 0, chosen_phi, False)

        self._last_features[battle_id] = chosen_phi
        
        return self.create_order(chosen_action)

    def battle_finished_callback(self, battle):
        battle_id = battle.battle_tag
        self.last_battle_won = battle.won
        
        if battle_id in self._last_features:
            last_phi = self._last_features[battle_id]
            
            # Win/Loss Reward
            reward = 1 if battle.won else -1
            
            # Terminal state
            self.memory.push(last_phi, 0, reward, np.zeros_like(last_phi), True)
            
            del self._last_features[battle_id]
            
            if battle_id in self._battles:
                del self._battles[battle_id]

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # FIX: Removed the extra variable "_" (action) from unpacking
        # The memory sample returns (state, reward, next_state, done)
        states, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q(s, a)
        current_q = self.model(states)
        
        # Next Q(s', a')
        with torch.no_grad():
            next_q = self.target_model(next_states)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Use nn.MSELoss (requires import torch.nn as nn)
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"DQN Model loaded from {path}")
        except Exception as e:
            print(f"No valid checkpoint found: {e}. Starting fresh.")