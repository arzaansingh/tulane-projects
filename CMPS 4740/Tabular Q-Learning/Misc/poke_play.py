import asyncio
import numpy as np
import pickle
import os
import uuid

# --- IMPORTS ---
from poke_env.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.battle import pokemon 
from poke_env.ps_client.account_configuration import AccountConfiguration

# ==================================================================================
# MONKEY PATCH: FIX GEN 1 CRASHES
# ==================================================================================
_original_available_moves = pokemon.Pokemon.available_moves_from_request

def patched_available_moves(self, request_moves):
    try:
        return _original_available_moves(self, request_moves)
    except AssertionError:
        return []

pokemon.Pokemon.available_moves_from_request = patched_available_moves
# ==================================================================================


class CustomBattleOrder:
    def __init__(self, action):
        self.action = action
    
    @property
    def message(self):
        # Duck Typing to detect Switch vs Move
        if hasattr(self.action, "species"):
            return f"/choose switch {self.action.species}"
        return f"/choose move {self.action.id}"


class QLearningPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", **kwargs):
        super().__init__(battle_format=battle_format, **kwargs)
        self.Q = {}
        # Zero randomness (Play to win)
        self.epsilon = 0.0 
    
    def load_model(self, filename="q_table_switch.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.Q = pickle.load(f)
                print(f"Brain loaded: {filename} ({len(self.Q)} states)")
            except Exception as e:
                print(f"Error loading brain: {e}")
                self.Q = {}
        else:
            print(f"Error: {filename} not found.")

    # --------------------- HELPER: GET ACTION KEY ---------------------
    def get_action_key(self, action):
        if hasattr(action, "species"):
            return f"switch_{action.species.lower()}" 
        return action.id 

    # --------------------- STATE (Must match training) ---------------------
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

    # --------------------- MOVE SELECTION ---------------------
    def choose_move(self, battle):
        current_state = self.get_state(battle)
        
        # Gather Actions
        available_moves = battle.available_moves
        available_switches = battle.available_switches
        possible_actions = []
        
        # Action Masking
        if available_moves:
            for move in available_moves:
                if move.id in ['recover', 'softboiled', 'rest', 'milkdrink'] and \
                   battle.active_pokemon.current_hp == battle.active_pokemon.max_hp: continue
                if move.category.name == 'STATUS' and battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
                     if move.target == 'normal' or move.target == 'any': continue
                possible_actions.append(move)
        
        if available_switches:
            for pokemon in available_switches:
                if not pokemon.fainted and not pokemon.active:
                    possible_actions.append(pokemon)

        if not possible_actions:
            return self.choose_random_move(battle)

        # Greedy Selection (Pick Best Q-Value)
        selected_action = None
        
        if current_state in self.Q:
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
        else:
            selected_action = random.choice(possible_actions)

        # LOGGING: Show what the bot is doing
        action_name = self.get_action_key(selected_action)
        print(f"Turn {battle.turn}: Bot chose {action_name}")
        
        return CustomBattleOrder(selected_action)


async def main():
    print("--- SETTING UP BOSS BATTLE ---")
    config = LocalhostServerConfiguration
    
    # Fixed Name so you can find it
    bot_username = "TheBossBot"
    
    player = QLearningPlayer(
        battle_format="gen1randombattle",
        server_configuration=config,
        account_configuration=AccountConfiguration(bot_username, None),
    )
    
    # Load the Switch-Enabled Brain
    player.load_model("q_table_switch.pkl")
    
    print(f"Bot is online: {bot_username}")
    print(f"Waiting for challenge on http://localhost:8000 ...")
    
    # Wait indefinitely for challenges
    await player.accept_challenges(None, n_challenges=10)

if __name__ == "__main__":
    asyncio.run(main())