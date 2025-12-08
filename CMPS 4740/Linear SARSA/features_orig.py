import numpy as np
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.status import Status

class FeatureExtractor:
    def __init__(self):
        self.types = [
            'Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 
            'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 
            'Bug', 'Rock', 'Ghost', 'Dragon'
        ]
        self.special_types = {'Fire', 'Water', 'Grass', 'Ice', 'Electric', 'Psychic', 'Dragon'}
        
        # Gen 1 Type Chart (Attacker -> Defender)
        self.gen1_chart = {
            ('Normal', 'Ghost'): 0.0, ('Normal', 'Rock'): 0.5,
            ('Fire', 'Fire'): 0.5, ('Fire', 'Water'): 0.5, ('Fire', 'Grass'): 2.0, ('Fire', 'Ice'): 2.0, ('Fire', 'Bug'): 2.0, ('Fire', 'Rock'): 0.5, ('Fire', 'Dragon'): 0.5,
            ('Water', 'Fire'): 2.0, ('Water', 'Water'): 0.5, ('Water', 'Grass'): 0.5, ('Water', 'Ground'): 2.0, ('Water', 'Rock'): 2.0, ('Water', 'Dragon'): 0.5,
            ('Electric', 'Water'): 2.0, ('Electric', 'Electric'): 0.5, ('Electric', 'Grass'): 0.5, ('Electric', 'Ground'): 0.0, ('Electric', 'Flying'): 2.0, ('Electric', 'Dragon'): 0.5,
            ('Grass', 'Fire'): 0.5, ('Grass', 'Water'): 2.0, ('Grass', 'Grass'): 0.5, ('Grass', 'Poison'): 0.5, ('Grass', 'Ground'): 2.0, ('Grass', 'Flying'): 0.5, ('Grass', 'Bug'): 0.5, ('Grass', 'Rock'): 2.0, ('Grass', 'Dragon'): 0.5,
            ('Ice', 'Water'): 0.5, ('Ice', 'Grass'): 2.0, ('Ice', 'Ice'): 0.5, ('Ice', 'Ground'): 2.0, ('Ice', 'Flying'): 2.0, ('Ice', 'Dragon'): 2.0,
            ('Fighting', 'Normal'): 2.0, ('Fighting', 'Ice'): 2.0, ('Fighting', 'Poison'): 0.5, ('Fighting', 'Flying'): 0.5, ('Fighting', 'Psychic'): 0.5, ('Fighting', 'Bug'): 0.5, ('Fighting', 'Rock'): 2.0, ('Fighting', 'Ghost'): 0.0,
            ('Poison', 'Grass'): 2.0, ('Poison', 'Poison'): 0.5, ('Poison', 'Ground'): 0.5, ('Poison', 'Bug'): 2.0, ('Poison', 'Rock'): 0.5, ('Poison', 'Ghost'): 0.5,
            ('Ground', 'Fire'): 2.0, ('Ground', 'Electric'): 2.0, ('Ground', 'Grass'): 0.5, ('Ground', 'Poison'): 2.0, ('Ground', 'Flying'): 0.0, ('Ground', 'Bug'): 0.5, ('Ground', 'Rock'): 2.0,
            ('Flying', 'Electric'): 0.5, ('Flying', 'Grass'): 2.0, ('Flying', 'Fighting'): 2.0, ('Flying', 'Bug'): 2.0, ('Flying', 'Rock'): 0.5,
            ('Psychic', 'Fighting'): 2.0, ('Psychic', 'Poison'): 2.0, ('Psychic', 'Psychic'): 0.5,
            ('Bug', 'Fire'): 0.5, ('Bug', 'Grass'): 2.0, ('Bug', 'Fighting'): 0.5, ('Bug', 'Poison'): 2.0, ('Bug', 'Flying'): 0.5, ('Bug', 'Ghost'): 0.5,
            ('Rock', 'Fire'): 2.0, ('Rock', 'Ice'): 2.0, ('Rock', 'Fighting'): 0.5, ('Rock', 'Ground'): 0.5, ('Rock', 'Flying'): 2.0, ('Rock', 'Bug'): 2.0,
            ('Ghost', 'Normal'): 0.0, ('Ghost', 'Psychic'): 0.0, ('Ghost', 'Ghost'): 2.0,
            ('Dragon', 'Dragon'): 2.0
        }

    # Helper method for the agent
    def get_effectiveness(self, move_type, def_type1, def_type2=None):
        eff = self.gen1_chart.get((move_type, def_type1), 1.0)
        if def_type2:
            eff *= self.gen1_chart.get((move_type, def_type2), 1.0)
        return eff

    def get_features(self, battle, move_obj=None):
        # --- STATE FEATURES (s) ---
        # 1. HP & Status (Dense)
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0
        
        my_status = 0.0
        if battle.active_pokemon and battle.active_pokemon.status:
            if battle.active_pokemon.status.name in ['SLP', 'FRZ']: my_status = 1.0
            else: my_status = 0.5
            
        opp_status = 0.0
        if battle.opponent_active_pokemon and battle.opponent_active_pokemon.status:
            if battle.opponent_active_pokemon.status.name in ['SLP', 'FRZ']: opp_status = 1.0
            else: opp_status = 0.5

        # 2. Speed Advantage (Dense)
        speed_adv = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            my_spe = battle.active_pokemon.base_stats['spe']
            opp_spe = battle.opponent_active_pokemon.base_stats['spe']
            if my_spe > opp_spe: speed_adv = 1.0
            elif my_spe < opp_spe: speed_adv = -1.0

        # 3. TYPE EMBEDDING (One-Hot)
        my_type_vec = np.zeros(len(self.types))
        opp_type_vec = np.zeros(len(self.types))
        
        if battle.active_pokemon:
            for t in battle.active_pokemon.types:
                if t.name in self.types: my_type_vec[self.types.index(t.name)] = 1.0
        
        if battle.opponent_active_pokemon:
            for t in battle.opponent_active_pokemon.types:
                if t.name in self.types: opp_type_vec[self.types.index(t.name)] = 1.0

        state_vec = np.concatenate([
            [my_hp, opp_hp, my_status, opp_status, speed_adv, 1.0], # 6
            my_type_vec, # 15
            opp_type_vec # 15
        ])

        if move_obj is None:
            return state_vec

        # --- ACTION FEATURES (a) ---
        action_vec = self._get_action_features(battle, move_obj)
        return np.concatenate([state_vec, action_vec])

    def _get_action_features(self, battle, move):
        # Action Semantic Vector
        is_switch = 0.0
        move_type_vec = np.zeros(len(self.types)) # One-Hot Move Type
        
        dmg_pot = 0.0
        accuracy = 0.0
        is_special = 0.0
        is_status_move = 0.0
        is_recovery = 0.0
        
        if not isinstance(move, Move):
            # SWITCH
            is_switch = 1.0
            if hasattr(move, 'types'):
                for t in move.types:
                    if t.name in self.types: move_type_vec[self.types.index(t.name)] = 1.0
        else:
            # MOVE
            if move.type.name in self.types:
                move_type_vec[self.types.index(move.type.name)] = 1.0
            
            if move.base_power > 0:
                stab = 1.5 if move.type in battle.active_pokemon.types else 1.0
                dmg_pot = (move.base_power * stab) / 250.0
            
            if move.accuracy is True: accuracy = 1.0
            else: accuracy = move.accuracy / 100.0
            
            if move.type.name in self.special_types: is_special = 1.0
            
            if move.status or move.volatile_status: is_status_move = 1.0
            if move.id in ['recover', 'softboiled', 'rest']: is_recovery = 1.0

        return np.concatenate([
            [is_switch, dmg_pot, accuracy, is_special, is_status_move, is_recovery],
            move_type_vec
        ])

    @property
    def total_dim(self):
        # State: 6 + 15 + 15 = 36
        # Action: 6 + 15 = 21
        # Total: 57
        return 57