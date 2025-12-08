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

# Store the original move validation function
_original_available_moves = pokemon.Pokemon.available_moves_from_request

# Override the move validation to prevent crashes in Gen 1 battles
def patched_available_moves(self, request_moves):
    try:
        return _original_available_moves(self, request_moves)
    except AssertionError:
        return []

# Apply the monkey patch to the library
pokemon.Pokemon.available_moves_from_request = patched_available_moves


class CustomBattleOrder:
    def __init__(self, move):
        self.move = move
    @property
    def message(self):
        return f"/choose move {self.move.id}"


class QLearningPlayer(Player):
    def __init__(self, battle_format="gen1randombattle", alpha=0.1, gamma=0.9, epsilon=1.0, **kwargs):
        super().__init__(battle_format=battle_format, max_concurrent_battles=1, **kwargs)
        self.Q = {}
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        self.previous_state = None
        self.previous_action = None

        # specific variables to track changes between turns for reward calculation
        self.last_my_hp = 1.0
        self.last_opp_hp = 1.0
        self.last_opp_fainted = 0
        self.last_my_fainted = 0
        self.last_opp_status = None

    # Save the Q table to a file using pickle
    def save_model(self, filename="q_table_no_switch.pkl"):
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.Q.copy(), f)
        except Exception as e:
            print(f"Save failed (ignoring): {e}")

    # Load the Q table from a file if it exists
    def load_model(self, filename="q_table_no_switch.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.Q = pickle.load(f)
                print(f"Loaded Q-table with {len(self.Q)} states.")
            except (EOFError, pickle.UnpicklingError):
                print("Saved model was corrupted/empty. Starting from scratch.")
                self.Q = {}
        else:
            print("No saved model found. Starting from scratch.")

    # transform the battle object into a simple state tuple
    def get_state(self, battle):
        # convert hp to a discrete value between 0 and 4
        def get_hp_bucket(pokemon):
            if not pokemon or pokemon.current_hp == 0: return 0
            return int(pokemon.current_hp / pokemon.max_hp * 4) + 1

        # get the primary type of the pokemon
        def get_type(pokemon):
            if not pokemon: return "none"
            return str(pokemon.type_1).split(".")[-1].lower()

        # group status conditions by severity
        def get_status_bucket(pokemon):
            if not pokemon or not pokemon.status: return 0
            s = str(pokemon.status).split('.')[-1].lower()
            if s in ['slp', 'frz']: return 2 
            if s in ['par', 'brn', 'psn', 'tox']: return 1 
            return 0

        # check if our active pokemon is faster than the opponent
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

    # calculate the numerical reward for the previous action
    def _compute_reward(self, battle):
        reward = 0.0
        
        # large reward for winning and penalty for losing
        if battle.won:
            return 100.0
        if battle.lost:
            return -100.0

        def get_hp_percent(pokemon):
            if not pokemon or pokemon.max_hp == 0: return 0.0
            return pokemon.current_hp / pokemon.max_hp

        my_current_hp = get_hp_percent(battle.active_pokemon)
        opp_current_hp = get_hp_percent(battle.opponent_active_pokemon)

        # reward dealing damage and punish taking damage
        damage_dealt = self.last_opp_hp - opp_current_hp
        damage_taken = self.last_my_hp - my_current_hp
        reward += (damage_dealt - damage_taken) * 50.0 

        # reward knocking out an opponent and punish losing a pokemon
        opp_fainted_count = len([mon for mon in battle.opponent_team.values() if mon.fainted])
        my_fainted_count = len([mon for mon in battle.team.values() if mon.fainted])

        if opp_fainted_count > self.last_opp_fainted:
            reward += 20.0
        if my_fainted_count > self.last_my_fainted:
            reward -= 20.0

        # small reward for inflicting a status condition
        opp_status = battle.opponent_active_pokemon.status if battle.opponent_active_pokemon else None
        if opp_status != self.last_opp_status and opp_status is not None:
            reward += 10.0 

        return reward

    # decide which move to use based on the q table
    def choose_move(self, battle):
        # observe current state and calculate reward from the last turn
        current_state = self.get_state(battle)
        reward = self._compute_reward(battle)
        
        # update the tracking variables for the next turn
        def get_hp_percent(pokemon):
            if not pokemon or pokemon.max_hp == 0: return 0.0
            return pokemon.current_hp / pokemon.max_hp
            
        self.last_my_hp = get_hp_percent(battle.active_pokemon)
        self.last_opp_hp = get_hp_percent(battle.opponent_active_pokemon)
        self.last_opp_fainted = len([mon for mon in battle.opponent_team.values() if mon.fainted])
        self.last_my_fainted = len([mon for mon in battle.team.values() if mon.fainted])
        self.last_opp_status = battle.opponent_active_pokemon.status if battle.opponent_active_pokemon else None

        # update the q value for the previous state and action
        if self.previous_state and self.previous_action:
            if self.previous_state not in self.Q: self.Q[self.previous_state] = {}
            if self.previous_action not in self.Q[self.previous_state]: self.Q[self.previous_state][self.previous_action] = 0.0
            
            max_q_current = 0.0
            if current_state in self.Q and self.Q[current_state]:
                max_q_current = max(self.Q[current_state].values())

            old_q = self.Q[self.previous_state][self.previous_action]
            new_q = old_q + self.alpha * (reward + self.gamma * max_q_current - old_q)
            self.Q[self.previous_state][self.previous_action] = new_q

        # filter out moves that are logically bad to speed up learning
        available_moves = battle.available_moves
        good_moves = []

        if available_moves:
            for move in available_moves:
                # do not heal if hp is full
                if move.id in ['recover', 'softboiled', 'rest', 'milkdrink'] and \
                   battle.active_pokemon.current_hp == battle.active_pokemon.max_hp:
                    continue
                
                # do not use status moves if the opponent already has a status
                if move.category.name == 'STATUS' and battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
                     if move.target == 'normal' or move.target == 'any': 
                         continue
                
                good_moves.append(move)
        
        actions = good_moves if good_moves else available_moves
        
        if not actions:
            return self.choose_random_move(battle)

        # use epsilon greedy strategy to choose action
        selected_move = None
        if random.random() < self.epsilon or current_state not in self.Q:
            selected_move = random.choice(actions)
        else:
            q_values = self.Q[current_state]
            if q_values:
                best_move_id = max(q_values, key=q_values.get)
                # ensure we pick from the filtered list of actions
                matching_moves = [m for m in actions if m.id == best_move_id]
                
                if matching_moves:
                    selected_move = matching_moves[0]
                else:
                    selected_move = random.choice(actions)
            else:
                selected_move = random.choice(actions)

        self.previous_state = current_state
        self.previous_action = selected_move.id
        return CustomBattleOrder(selected_move)
    
    # reset tracking variables when a battle ends
    async def battle_end_callback(self, battle):
        self.previous_state = None
        self.previous_action = None
        self.last_my_hp = 1.0
        self.last_opp_hp = 1.0
        self.last_opp_fainted = 0
        self.last_my_fainted = 0
        self.last_opp_status = None


async def main():
    print("Creating Gen 1 players...")
    config = LocalhostServerConfiguration

    # create unique names for the bots
    p1_name = f"QBot_{str(uuid.uuid4())[:8]}"
    p2_name = f"MaxBot_{str(uuid.uuid4())[:8]}"

    p1 = QLearningPlayer(
        battle_format="gen1randombattle", 
        server_configuration=config,
        account_configuration=AccountConfiguration(p1_name, None),
        epsilon=0.1
    )
    
    p1.load_model("q_table_no_switch.pkl") 

    p2 = MaxBasePowerPlayer(
        battle_format="gen1randombattle",
        server_configuration=config,
        account_configuration=AccountConfiguration(p2_name, None),
        max_concurrent_battles=1
    )

    p1._start_challenging = True
    p2._start_challenging = False

    # set training parameters
    TOTAL_BATTLES = 10000  
    BATCH_SIZE = 100 
    
    # set linear decay parameters for epsilon
    MIN_EPSILON = 0.05
    EXPLORATION_PHASE = 0.8 
    
    battles_to_decay = TOTAL_BATTLES * EXPLORATION_PHASE
    epsilon_decay_amount = (1.0 - MIN_EPSILON) / battles_to_decay
    
    x_axis = []  
    y_axis = []  
    prev_win_count = p1.n_won_battles

    print(f"Starting training loop for {TOTAL_BATTLES} battles...")
    print(f"Linear Decay: Epsilon will hit {MIN_EPSILON} at battle {int(battles_to_decay)}.")
    
    for i in range(TOTAL_BATTLES):
        try:
            await p1.battle_against(p2, n_battles=1)
            
            # decrease epsilon after every battle
            if p1.epsilon > MIN_EPSILON:
                p1.epsilon -= epsilon_decay_amount
            else:
                p1.epsilon = MIN_EPSILON

        except Exception as e:
            print(f"Battle {i+1} failed. Continuing...")
            continue

        # log progress and save model every batch
        if (i + 1) % BATCH_SIZE == 0:
            p1.save_model("q_table_no_switch.pkl")
            
            current_win_count = p1.n_won_battles
            wins_in_batch = current_win_count - prev_win_count
            win_rate_batch = wins_in_batch / BATCH_SIZE
            
            x_axis.append(i + 1)
            y_axis.append(win_rate_batch)
            prev_win_count = current_win_count
            
            print(f"Batch {i+1}: Rolling Win Rate {win_rate_batch:.2%} | Epsilon: {p1.epsilon:.4f}")

    print("Training finished.")
    p1.save_model("q_table_no_switch.pkl")
    
    # plot the learning curve
    try:
        print("Generating learning curve...")
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, y_axis, marker='o', linestyle='-', color='b', label='Win Rate (per 100 games)')
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% Win Rate') 
        plt.title(f'Q-Learning Agent Performance (Gen 1 Random)')
        plt.xlabel('Number of Battles')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1.0) 
        plt.grid(True)
        plt.legend()
        plt.savefig("learning_curve.png")
        print("Graph saved as 'learning_curve.png'. Check your folder!")
        
    except Exception as e:
        print(f"Could not plot graph: {e}")


if __name__ == "__main__":
    asyncio.run(main())