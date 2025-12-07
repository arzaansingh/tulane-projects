import numpy as np
import zlib

class DiscreteFeatureExtractor:
    def __init__(self):
        # We can hash strings to keep the tuple compact, 
        # or just use the move ID strings directly since tuples of strings are hashable in Python.
        # Strings are safer for debugging.
        pass

    def get_hp_bucket(self, current_hp, max_hp):
        if max_hp == 0: return 0
        ratio = current_hp / max_hp
        if ratio > 0.5: return 2 # Green
        if ratio > 0.2: return 1 # Yellow
        return 0 # Red

    def get_state_key(self, battle):
        # 1. My Active Pokemon
        my_mon = battle.active_pokemon
        my_species = my_mon.species if my_mon else "None"
        my_hp = self.get_hp_bucket(my_mon.current_hp, my_mon.max_hp) if my_mon else 0
        
        # 2. Opponent Active Pokemon
        opp_mon = battle.opponent_active_pokemon
        opp_species = opp_mon.species if opp_mon else "None"
        opp_hp = self.get_hp_bucket(opp_mon.current_hp, opp_mon.max_hp) if opp_mon else 0
        
        # 3. Moveset (The Core Change)
        # We take all known moves, sort them, and tuple them.
        # This makes the state permutation invariant.
        # (Thunderbolt, Surf) is the same state as (Surf, Thunderbolt)
        move_ids = []
        if my_mon:
            # use keys() from moves dict which contains all known moves
            move_ids = sorted(list(my_mon.moves.keys()))
            
        # Pad to 4 so the tuple size is constant
        while len(move_ids) < 4:
            move_ids.append("none")
            
        moves_tuple = tuple(move_ids[:4])

        # 4. Create Tuple Key
        # (MySpecies, MyHP, OppSpecies, OppHP, Move1, Move2, Move3, Move4)
        state_key = (my_species, my_hp, opp_species, opp_hp, moves_tuple)
        
        return state_key