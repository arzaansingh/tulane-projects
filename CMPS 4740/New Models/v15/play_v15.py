import asyncio
import os
import logging
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from player_v15 import TabularQPlayerV15

# 1. SILENCE THE NOISE
# This stops poke-env from printing websocket messages and format lists
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)

async def main():
    # 2. Initialize the agent for GEN 1
    agent = TabularQPlayerV15(
        battle_format="gen1randombattle", # Changed to Gen 1
        server_configuration=LocalhostServerConfiguration,
        epsilon=0.0,  # Play optimally (Greedy)
        max_concurrent_battles=1
    )

    # 3. Load the trained table
    # Ensure this matches the file you actually trained (e.g., maxbp, random)
    TABLE_PATH = "v15_models/qtable_maxbp.pkl" 
    
    if os.path.exists(TABLE_PATH):
        agent.load_table(TABLE_PATH)
        # We print this manually so it looks clean
        print(f"\n✅ LOADED TABLES from {TABLE_PATH}")
        print(f"   - Master Q-Table Size: {len(agent.q_table)}")
        print(f"   - Switch Q-Table Size: {len(agent.switch_table)}")
    else:
        print(f"\n⚠️ WARNING: Could not find {TABLE_PATH}.")
        print("   Agent will play with a completely random (empty) brain.")

    # 4. Print the Username and Instructions
    print("\n" + "="*50)
    print(f"🤖 AGENT ONLINE:  {agent.username}")
    print("="*50)
    print(f"👉 Open your browser to: http://localhost:8000")
    print(f"👉 Search for user '{agent.username}'")
    print(f"👉 Challenge to: [Gen 1] Random Battle")
    print("="*50 + "\n")

    # Accept 100 challenges
    await agent.accept_challenges(None, 100)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Agent shutting down. Thanks for playing!")