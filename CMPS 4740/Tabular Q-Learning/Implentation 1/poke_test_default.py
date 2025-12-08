from poke_env.player import RandomPlayer

def main():
    print("Creating Gen 1 players...")

    p1 = RandomPlayer(battle_format="gen1randombattle", max_concurrent_battles=1)
    p2 = RandomPlayer(battle_format="gen1randombattle", max_concurrent_battles=1)

    print("Starting 3 Gen 1 battles inside the simulator...")

    # Classic poke-env loop for player-vs-player
    for i in range(3):
        battle = p1.battle_against(p2, n_battles=1)
        print(f"Battle {i+1} result: {p1.n_won_battles} wins for p1")

    print("All battles finished.")

if __name__ == "__main__":
    main()
