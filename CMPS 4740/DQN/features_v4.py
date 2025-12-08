import numpy as np
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.status import Status

class FeatureExtractor:
    def __init__(self):
        # NORMALIZED TYPES (ALL UPPERCASE)
        self.types = [
            'NORMAL', 'FIRE', 'WATER', 'ELECTRIC', 'GRASS', 'ICE', 
            'FIGHTING', 'POISON', 'GROUND', 'FLYING', 'PSYCHIC', 
            'BUG', 'ROCK', 'GHOST', 'DRAGON'
        ]
        self.special_types = {'FIRE', 'WATER', 'GRASS', 'ICE', 'ELECTRIC', 'PSYCHIC', 'DRAGON'}
        
        # GEN 1 TYPE CHART (Upper Case Keys)
        self.gen1_chart = {
            ('NORMAL', 'GHOST'): 0.0, ('NORMAL', 'ROCK'): 0.5,
            ('FIRE', 'FIRE'): 0.5, ('FIRE', 'WATER'): 0.5, ('FIRE', 'GRASS'): 2.0, ('FIRE', 'ICE'): 2.0, ('FIRE', 'BUG'): 2.0, ('FIRE', 'ROCK'): 0.5, ('FIRE', 'DRAGON'): 0.5,
            ('WATER', 'FIRE'): 2.0, ('WATER', 'WATER'): 0.5, ('WATER', 'GRASS'): 0.5, ('WATER', 'GROUND'): 2.0, ('WATER', 'ROCK'): 2.0, ('WATER', 'DRAGON'): 0.5,
            ('ELECTRIC', 'WATER'): 2.0, ('ELECTRIC', 'ELECTRIC'): 0.5, ('ELECTRIC', 'GRASS'): 0.5, ('ELECTRIC', 'GROUND'): 0.0, ('ELECTRIC', 'FLYING'): 2.0, ('ELECTRIC', 'DRAGON'): 0.5,
            ('GRASS', 'FIRE'): 0.5, ('GRASS', 'WATER'): 2.0, ('GRASS', 'GRASS'): 0.5, ('GRASS', 'POISON'): 0.5, ('GRASS', 'GROUND'): 2.0, ('GRASS', 'FLYING'): 0.5, ('GRASS', 'BUG'): 0.5, ('GRASS', 'ROCK'): 2.0, ('GRASS', 'DRAGON'): 0.5,
            ('ICE', 'WATER'): 0.5, ('ICE', 'GRASS'): 2.0, ('ICE', 'ICE'): 0.5, ('ICE', 'GROUND'): 2.0, ('ICE', 'FLYING'): 2.0, ('ICE', 'DRAGON'): 2.0,
            ('FIGHTING', 'NORMAL'): 2.0, ('FIGHTING', 'ICE'): 2.0, ('FIGHTING', 'POISON'): 0.5, ('FIGHTING', 'FLYING'): 0.5, ('FIGHTING', 'PSYCHIC'): 0.5, ('FIGHTING', 'BUG'): 0.5, ('FIGHTING', 'ROCK'): 2.0, ('FIGHTING', 'GHOST'): 0.0,
            ('POISON', 'GRASS'): 2.0, ('POISON', 'POISON'): 0.5, ('POISON', 'GROUND'): 0.5, ('POISON', 'BUG'): 2.0, ('POISON', 'ROCK'): 0.5, ('POISON', 'GHOST'): 0.5,
            ('GROUND', 'FIRE'): 2.0, ('GROUND', 'ELECTRIC'): 2.0, ('GROUND', 'GRASS'): 0.5, ('GROUND', 'POISON'): 2.0, ('GROUND', 'FLYING'): 0.0, ('GROUND', 'BUG'): 0.5, ('GROUND', 'ROCK'): 2.0,
            ('FLYING', 'ELECTRIC'): 0.5, ('FLYING', 'GRASS'): 2.0, ('FLYING', 'FIGHTING'): 2.0, ('FLYING', 'BUG'): 2.0, ('FLYING', 'ROCK'): 0.5,
            ('PSYCHIC', 'FIGHTING'): 2.0, ('PSYCHIC', 'POISON'): 2.0, ('PSYCHIC', 'PSYCHIC'): 0.5,
            ('BUG', 'FIRE'): 0.5, ('BUG', 'GRASS'): 2.0, ('BUG', 'FIGHTING'): 0.5, ('BUG', 'POISON'): 2.0, ('BUG', 'FLYING'): 0.5, ('BUG', 'GHOST'): 0.5,
            ('ROCK', 'FIRE'): 2.0, ('ROCK', 'ICE'): 2.0, ('ROCK', 'FIGHTING'): 0.5, ('ROCK', 'GROUND'): 0.5, ('ROCK', 'FLYING'): 2.0, ('ROCK', 'BUG'): 2.0,
            ('GHOST', 'NORMAL'): 0.0, ('GHOST', 'PSYCHIC'): 0.0, ('GHOST', 'GHOST'): 2.0,
            ('DRAGON', 'DRAGON'): 2.0
        }

    def get_effectiveness(self, move_type, def_type1, def_type2=None):
        if not move_type or not def_type1: return 1.0
        # Force UPPERCASE matching
        m_t = move_type.upper()
        d_t1 = def_type1.upper()
        eff = self.gen1_chart.get((m_t, d_t1), 1.0)
        if def_type2:
            d_t2 = def_type2.upper()
            eff *= self.gen1_chart.get((m_t, d_t2), 1.0)
        return eff

    def get_features(self, battle, move_obj=None):
        # --- STATE FEATURES (13) ---
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

        speed_adv = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            my_spe = battle.active_pokemon.base_stats['spe']
            opp_spe = battle.opponent_active_pokemon.base_stats['spe']
            if my_spe > opp_spe: speed_adv = 1.0
            elif my_spe < opp_spe: speed_adv = -1.0

        def_matchup = 0.0
        if battle.active_pokemon and battle.opponent_active_pokemon:
            opp_t1 = battle.opponent_active_pokemon.type_1.name
            my_t1 = battle.active_pokemon.type_1.name
            my_t2 = battle.active_pokemon.type_2.name if battle.active_pokemon.type_2 else None
            eff = self.get_effectiveness(opp_t1, my_t1, my_t2)
            def_matchup = min(eff / 4.0, 1.0)

        # Stats
        my_boosts = np.zeros(3)
        if battle.active_pokemon:
            my_boosts[0] = battle.active_pokemon.boosts.get('atk', 0) / 6.0
            my_boosts[1] = battle.active_pokemon.boosts.get('spa', 0) / 6.0
            my_boosts[2] = battle.active_pokemon.boosts.get('spe', 0) / 6.0

        opp_boosts = np.zeros(3)
        if battle.opponent_active_pokemon:
            opp_boosts[0] = battle.opponent_active_pokemon.boosts.get('atk', 0) / 6.0
            opp_boosts[1] = battle.opponent_active_pokemon.boosts.get('spa', 0) / 6.0
            opp_boosts[2] = battle.opponent_active_pokemon.boosts.get('spe', 0) / 6.0

        state_vec = np.concatenate([
            [my_hp, opp_hp, my_status, opp_status, speed_adv, def_matchup], 
            my_boosts,
            opp_boosts,
            [1.0] # Bias
        ])

        if move_obj is None:
            return state_vec

        # --- ACTION FEATURES (8) ---
        action_vec = self._get_action_features(battle, move_obj)
        return np.concatenate([state_vec, action_vec])

    def _get_action_features(self, battle, move):
        is_move = 0.0
        is_switch = 0.0
        
        dmg_pot = 0.0
        accuracy = 0.0
        is_status = 0.0
        is_recovery = 0.0
        switch_def_adv = 0.0
        is_stab = 0.0

        if isinstance(move, Move):
            is_move = 1.0
            
            if move.base_power > 0:
                bp = move.base_power
                if move.type in battle.active_pokemon.types: 
                    bp *= 1.5
                    is_stab = 1.0
                
                eff = 1.0
                if battle.opponent_active_pokemon:
                    t1 = battle.opponent_active_pokemon.type_1.name
                    t2 = battle.opponent_active_pokemon.type_2.name if battle.opponent_active_pokemon.type_2 else None
                    eff = self.get_effectiveness(move.type.name, t1, t2)
                
                dmg_pot = (bp * eff) / 300.0
            
            if move.accuracy is True: accuracy = 1.0
            else: accuracy = move.accuracy / 100.0
            
            if move.status or move.volatile_status: is_status = 1.0
            if move.id in ['recover', 'softboiled', 'rest']: is_recovery = 1.0

        else:
            is_switch = 1.0
            if battle.opponent_active_pokemon:
                opp_type = battle.opponent_active_pokemon.type_1.name
                my_t1 = move.type_1.name
                my_t2 = move.type_2.name if move.type_2 else None
                
                eff = self.get_effectiveness(opp_type, my_t1, my_t2)
                if eff == 0.0: switch_def_adv = 1.0 
                elif eff < 1.0: switch_def_adv = 0.5 
                elif eff > 1.0: switch_def_adv = -0.5 

        return np.array([
            is_move, is_switch,
            dmg_pot, accuracy, is_stab,
            is_status, is_recovery, switch_def_adv
        ])

    @property
    def total_dim(self):
        return 21