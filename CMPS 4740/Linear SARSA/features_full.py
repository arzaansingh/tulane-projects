import numpy as np
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.effect import Effect
from poke_env.battle.status import Status

class FeatureExtractor:
    def __init__(self):
        # --- GEN 1 DATABASE (151 Pokemon) ---
        self.pokedex = [
            'bulbasaur', 'ivysaur', 'venusaur', 'charmander', 'charmeleon', 'charizard',
            'squirtle', 'wartortle', 'blastoise', 'caterpie', 'metapod', 'butterfree',
            'weedle', 'kakuna', 'beedrill', 'pidgey', 'pidgeotto', 'pidgeot', 'rattata',
            'raticate', 'spearow', 'fearow', 'ekans', 'arbok', 'pikachu', 'raichu',
            'sandshrew', 'sandslash', 'nidoranf', 'nidorina', 'nidoqueen', 'nidoranm',
            'nidorino', 'nidoking', 'clefairy', 'clefable', 'vulpix', 'ninetales',
            'jigglypuff', 'wigglytuff', 'zubat', 'golbat', 'oddish', 'gloom', 'vileplume',
            'paras', 'parasect', 'venonat', 'venomoth', 'diglett', 'dugtrio', 'meowth',
            'persian', 'psyduck', 'golduck', 'mankey', 'primeape', 'growlithe', 'arcanine',
            'poliwag', 'poliwhirl', 'poliwrath', 'abra', 'kadabra', 'alakazam', 'machop',
            'machoke', 'machamp', 'bellsprout', 'weepinbell', 'victreebel', 'tentacool',
            'tentacruel', 'geodude', 'graveler', 'golem', 'ponyta', 'rapidash', 'slowpoke',
            'slowbro', 'magnemite', 'magneton', 'farfetchd', 'doduo', 'dodrio', 'seel',
            'dewgong', 'grimer', 'muk', 'shellder', 'cloyster', 'gastly', 'haunter', 'gengar',
            'onix', 'drowzee', 'hypno', 'krabby', 'kingler', 'voltorb', 'electrode',
            'exeggcute', 'exeggutor', 'cubone', 'marowak', 'hitmonlee', 'hitmonchan',
            'lickitung', 'koffing', 'weezing', 'rhyhorn', 'rhydon', 'chansey', 'tangela',
            'kangaskhan', 'horsea', 'seadra', 'goldeen', 'seaking', 'staryu', 'starmie',
            'mrmime', 'scyther', 'jynx', 'electabuzz', 'magmar', 'pinsir', 'tauros',
            'magikarp', 'gyarados', 'lapras', 'ditto', 'eevee', 'vaporeon', 'jolteon',
            'flareon', 'porygon', 'omanyte', 'omastar', 'kabuto', 'kabutops', 'aerodactyl',
            'snorlax', 'articuno', 'zapdos', 'moltres', 'dratini', 'dragonair', 'dragonite',
            'mewtwo', 'mew'
        ]

        self.moves = [
            'bodyslam', 'hyperbeam', 'earthquake', 'blizzard', 'thunderbolt', 'psychic',
            'softboiled', 'recover', 'thunderwave', 'sleeppowder', 'explosion', 'selfdestruct',
            'surf', 'icebeam', 'drillpeck', 'rockslide', 'megadrain', 'rest', 'amnesia',
            'agility', 'swordsdance', 'wrap', 'seismictoss', 'nightshade', 'substitute',
            'reflect', 'stunspore', 'lovelykiss', 'sing', 'hypnosis', 'fireblast',
            'hydropump', 'razorleaf', 'slash', 'thunder', 'toxic', 'clamp', 'bind',
            'firespin', 'confuseray', 'doubleteam', 'bubblebeam', 'aurorabeam', 'crabhammer',
            'skyattack', 'transform', 'mimic', 'metronome', 'splash', 'struggle',
            'quickattack', 'doublekick', 'submission', 'counter', 'strength', 'fly',
            'dig', 'teleport', 'leechseed', 'glare', 'supersonic', 'barrier', 'lightscreen',
            'haze', 'screech', 'growth', 'sharpen', 'defensecurl', 'meditate', 'doubleedge',
            'bubble', 'watergun', 'ember', 'flamethrower', 'psywave', 'swift', 'triattack',
            'headbutt', 'stomp', 'lowkick', 'jumpkick', 'rollingkick', 'pinmissile',
            'acid', 'acidarmor', 'aerialace', 'aeroblast', 'absorb', 'auroraveil', 
            'bite', 'bonemerang', 'boneclub', 'constrict', 'conversion', 'cut',
            'dizzypunch', 'dreameater', 'eggbomb', 'fissure', 'guillotine', 'horndrill',
            'karatechop', 'kinesis', 'leechlife', 'lick', 'lovelykiss', 'lowsweep',
            'megakick', 'megapunch', 'minimize', 'mirrormove', 'mist', 'payday',
            'peck', 'petaldance', 'poisonpowder', 'pound', 'psybeam', 'rage',
            'razorwind', 'roar', 'rockthrow', 'sandattack', 'scratch', 'slam',
            'smokescreen', 'softboiled', 'solarbeam', 'sonicboom', 'spikecannon',
            'spore', 'stringshot', 'superfang', 'supersonic', 'swift', 'swordsdance',
            'tackle', 'tailwhip', 'takedown', 'teleport', 'thrash', 'thunderpunch',
            'thundershock', 'twineedle', 'vicegrip', 'vinewhip', 'waterfall',
            'whirlwind', 'wingattack', 'withdraw'
        ]
        
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
        self.special_types = {'Fire', 'Water', 'Grass', 'Ice', 'Electric', 'Psychic', 'Dragon'}

    def get_effectiveness(self, move_type, def_type1, def_type2=None):
        eff = self.gen1_chart.get((move_type, def_type1), 1.0)
        if def_type2:
            eff *= self.gen1_chart.get((move_type, def_type2), 1.0)
        return eff

    def get_features(self, battle, move_obj=None):
        state_parts = []
        
        def get_mon_features(mon):
            hp = mon.current_hp_fraction
            
            stats = mon.base_stats
            boosts = mon.boosts
            
            stat_vec = np.array([
                stats.get('atk', 100)/255.0, stats.get('def', 100)/255.0, 
                stats.get('spa', 100)/255.0, stats.get('spe', 100)/255.0
            ])
            boost_vec = np.array([
                boosts.get('atk', 0)/6.0, boosts.get('def', 0)/6.0, 
                boosts.get('spa', 0)/6.0, boosts.get('spe', 0)/6.0,
                boosts.get('accuracy', 0)/6.0, boosts.get('evasion', 0)/6.0
            ])
            
            status_vec = np.zeros(8)
            if mon.fainted: status_vec[6] = 1.0
            elif mon.status is None: status_vec[7] = 1.0
            else:
                s_map = {Status.SLP:0, Status.PSN:1, Status.BRN:2, Status.FRZ:3, Status.PAR:4, Status.TOX:5}
                if mon.status in s_map: status_vec[s_map[mon.status]] = 1.0

            vol_vec = np.zeros(5) 
            if Effect.CONFUSION in mon.effects: vol_vec[0] = 1.0
            if Effect.SUBSTITUTE in mon.effects: vol_vec[1] = 1.0
            if Effect.LEECH_SEED in mon.effects: vol_vec[2] = 1.0
            
            species_vec = np.zeros(len(self.pokedex))
            s_name = mon.species.lower()
            if s_name in self.pokedex: species_vec[self.pokedex.index(s_name)] = 1.0
            
            return np.concatenate([[hp], stat_vec, boost_vec, status_vec, vol_vec, species_vec])

        if battle.active_pokemon:
            state_parts.append(get_mon_features(battle.active_pokemon))
        else:
            state_parts.append(np.zeros(1 + 4 + 6 + 8 + 5 + 151)) 
            
        if battle.opponent_active_pokemon:
            state_parts.append(get_mon_features(battle.opponent_active_pokemon))
        else:
            state_parts.append(np.zeros(1 + 4 + 6 + 8 + 5 + 151)) 

        side_vec = np.zeros(4) 
        if 'reflect' in battle.side_conditions: side_vec[0] = 1.0
        if 'lightscreen' in battle.side_conditions: side_vec[1] = 1.0
        if 'reflect' in battle.opponent_side_conditions: side_vec[2] = 1.0
        if 'lightscreen' in battle.opponent_side_conditions: side_vec[3] = 1.0
        
        state_parts.append(side_vec)
        state_parts.append([1.0]) 
        
        state_vec = np.concatenate(state_parts)

        if move_obj is None:
            return state_vec

        action_vec = self._get_action_features(battle, move_obj)
        return np.concatenate([state_vec, action_vec])

    def _get_action_features(self, battle, move):
        move_id_vec = np.zeros(len(self.moves))
        switch_id_vec = np.zeros(len(self.pokedex))
        
        is_switch = 0.0
        dmg_pot = 0.0
        accuracy = 0.0
        stat_aligned = 0.0 
        is_stab = 0.0
        is_status_move = 0.0
        is_recovery = 0.0
        switch_def_adv = 0.0

        if not isinstance(move, Move):
            is_switch = 1.0
            
            s_name = move.species.lower()
            if s_name in self.pokedex: switch_id_vec[self.pokedex.index(s_name)] = 1.0

            if battle.opponent_active_pokemon:
                opp_type = battle.opponent_active_pokemon.type_1.name
                my_t1 = move.type_1.name
                my_t2 = move.type_2.name if move.type_2 else None
                def_eff = self.get_effectiveness(opp_type, my_t1, my_t2)
                if def_eff < 1.0: switch_def_adv = 1.0 
                elif def_eff > 1.0: switch_def_adv = -1.0
        else:
            m_id = move.id
            if m_id in self.moves:
                move_id_vec[self.moves.index(m_id)] = 1.0
            
            if move.base_power > 0:
                bp = move.base_power
                if move.type in battle.active_pokemon.types: bp *= 1.5
                
                eff = 1.0
                if battle.opponent_active_pokemon:
                    t1 = battle.opponent_active_pokemon.type_1.name
                    t2 = battle.opponent_active_pokemon.type_2.name if battle.opponent_active_pokemon.type_2 else None
                    eff = self.get_effectiveness(move.type.name, t1, t2)
                
                dmg_pot = (bp * eff) / 300.0 

            if move.accuracy is True: accuracy = 1.0
            else: accuracy = move.accuracy / 100.0

            move_category = "Special" if move.type.name in self.special_types else "Physical"
            my_atk = battle.active_pokemon.base_stats['atk']
            my_spa = battle.active_pokemon.base_stats['spa']
            if move_category == "Physical" and my_atk >= my_spa: stat_aligned = 1.0
            elif move_category == "Special" and my_spa >= my_atk: stat_aligned = 1.0
            
            if move.status is not None or move.volatile_status is not None:
                is_status_move = 1.0
            if move.id in ['recover', 'softboiled', 'rest']:
                is_recovery = 1.0

        return np.concatenate([
            [is_switch, dmg_pot, accuracy, stat_aligned, is_stab, is_status_move, is_recovery, switch_def_adv],
            move_id_vec,
            switch_id_vec
        ])

    @property
    def total_dim(self):
        mon_size = 1 + 4 + 6 + 8 + 5 + len(self.pokedex) 
        state_size = (mon_size * 2) + 4 + 1 
        action_size = 8 + len(self.moves) + len(self.pokedex) 
        return state_size + action_size