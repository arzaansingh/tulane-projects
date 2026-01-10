import zlib
import logging

# Safety Imports
try:
    from poke_env.battle.status import Status
except ImportError:
    try:
        from poke_env.battle import Status
    except ImportError:
        Status = None

try:
    from poke_env.battle.side_condition import SideCondition
except ImportError:
    try:
        from poke_env.battle import SideCondition
    except ImportError:
        SideCondition = None

class AdvancedFeatureExtractor:
    def __init__(self):
        if Status is None:
            logging.warning("⚠️ Status enum not found. Features will be 0.")
        if SideCondition is None:
            logging.warning("⚠️ SideCondition enum not found. Features will be 0.")

    def get_hp_bucket(self, current_hp, max_hp):
        if max_hp == 0 or current_hp == 0: return 0
        ratio = current_hp / max_hp
        if ratio > 0.5: return 2 # High
        if ratio > 0.2: return 1 # Mid
        return 0 # Low

    def get_status_int(self, status):
        if status is None or Status is None: return 0
        return status.value

    def get_ability_hash(self, ability):
        if ability is None: return 0
        return zlib.adler32(ability.encode())

    def get_speed_check(self, my_mon, opp_mon):
        """
        Returns 1 if my_mon is faster than opp_mon.
        For active pokemon, uses current speed (with boosts).
        For benched pokemon, relies on base stats as best estimate.
        """
        if not my_mon or not opp_mon: return 0
        try:
            # Try to get volatile speed (if active), else base speed
            my_speed = getattr(my_mon, "speed", my_mon.base_stats.get('spe', 0))
            opp_speed = getattr(opp_mon, "speed", opp_mon.base_stats.get('spe', 0))
            return 1 if my_speed > opp_speed else 0
        except AttributeError:
            return 0

    def get_boost_flags(self, mon):
        """
        Returns a tuple (has_any_boost, has_max_boost).
        Checks 'atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'.
        """
        if not mon: return (0, 0)
        
        # mon.boosts is a dict like {'atk': 2, 'def': -1}
        boosts = mon.boosts
        
        has_any = 0
        has_max = 0
        
        for stat, stage in boosts.items():
            if stage > 0:
                has_any = 1
            if stage == 6:
                has_max = 1
                
        return (has_any, has_max)

    def get_hazards_tuple(self, battle):
        if SideCondition is None: return (0, 0, 0, 0)
        sc = battle.side_conditions
        has_spikes = 1 if SideCondition.SPIKES in sc else 0
        has_rocks = 1 if SideCondition.STEALTH_ROCK in sc else 0
        has_web = 1 if SideCondition.STICKY_WEB in sc else 0
        has_tspikes = 1 if SideCondition.TOXIC_SPIKES in sc else 0
        return (has_spikes, has_rocks, has_web, has_tspikes)

    def get_master_state(self, battle):
        # 1. My Active Pokemon
        my_mon = battle.active_pokemon
        if my_mon:
            my_species = my_mon.species
            my_hp = self.get_hp_bucket(my_mon.current_hp, my_mon.max_hp)
            my_status = self.get_status_int(my_mon.status)
            my_ability = self.get_ability_hash(my_mon.ability)
            # New Boost Features
            my_boosted, my_max_boosted = self.get_boost_flags(my_mon)
        else:
            my_species, my_hp, my_status, my_ability = "None", 0, 0, 0
            my_boosted, my_max_boosted = 0, 0
        
        # 2. Opponent Active Pokemon
        opp_mon = battle.opponent_active_pokemon
        if opp_mon:
            opp_species = opp_mon.species
            opp_hp = self.get_hp_bucket(opp_mon.current_hp, opp_mon.max_hp)
            opp_status = self.get_status_int(opp_mon.status)
            # We generally don't know opp ability or exact boosts as reliably without tracking
            # but we can check visible boosts if poke-env tracks them (it usually does)
            opp_boosted, opp_max_boosted = self.get_boost_flags(opp_mon)
        else:
            opp_species, opp_hp, opp_status = "None", 0, 0
            opp_boosted, opp_max_boosted = 0, 0

        # 3. Comparative Features
        is_faster = self.get_speed_check(my_mon, opp_mon)
        
        # Master Key
        return (
            my_species, my_hp, my_status, my_ability, my_boosted, my_max_boosted, is_faster, 
            opp_species, opp_hp, opp_status, opp_boosted, opp_max_boosted
        )

    def get_sub_state(self, battle, candidate):
        # 1. Opponent Context
        opp_mon = battle.opponent_active_pokemon
        if opp_mon:
            opp_species = opp_mon.species
            opp_hp = self.get_hp_bucket(opp_mon.current_hp, opp_mon.max_hp)
            opp_status = self.get_status_int(opp_mon.status)
        else:
            opp_species, opp_hp, opp_status = "None", 0, 0
            
        # 2. Candidate Context
        cand_species = candidate.species
        cand_hp = self.get_hp_bucket(candidate.current_hp, candidate.max_hp)
        cand_status = self.get_status_int(candidate.status)
        
        # 3. Hazards
        hazards = self.get_hazards_tuple(battle)

        # 4. Comparative Speed (NEW)
        # Check if the CANDIDATE (candidate) is faster than OPPONENT (opp_mon)
        is_faster = self.get_speed_check(candidate, opp_mon)
        
        # Sub Agent Key
        return (opp_species, opp_hp, opp_status, cand_species, cand_hp, cand_status, hazards, is_faster)