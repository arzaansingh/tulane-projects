import numpy as np

class DiscreteFeatureExtractor:
    def __init__(self):
        pass

    def get_hp_bucket(self, current_hp, max_hp):
        if max_hp == 0: return 0
        ratio = current_hp / max_hp
        if ratio > 0.5: return 2 # Green (High)
        if ratio > 0.2: return 1 # Yellow (Mid)
        return 0 # Red (Low)

    def get_state_key(self, battle):
        # 1. My Active Pokemon
        my_mon = battle.active_pokemon
        my_species = my_mon.species if my_mon else "None"
        my_hp = self.get_hp_bucket(my_mon.current_hp, my_mon.max_hp) if my_mon else 0
        
        # 2. Opponent Active Pokemon
        opp_mon = battle.opponent_active_pokemon
        opp_species = opp_mon.species if opp_mon else "None"
        opp_hp = self.get_hp_bucket(opp_mon.current_hp, opp_mon.max_hp) if opp_mon else 0
        
        # State Key: (MySpecies, MyHP, OppSpecies, OppHP)
        state_key = (my_species, my_hp, opp_species, opp_hp)
        
        return state_key