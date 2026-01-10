import asyncio
import os
import logging
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from player_v16 import TabularQPlayerV16

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("poke_env").setLevel(logging.CRITICAL)

async def main():
    agent = TabularQPlayerV16(
        battle_format="gen1randombattle", 
        server_configuration=LocalhostServerConfiguration,
        epsilon=0.0, 
        max_concurrent_battles=1
    )

    TABLE_PATH = "v16_models/qtable_maxbp.pkl" 
    
    if os.path.exists(TABLE_PATH):
        agent.load_table(TABLE_PATH)
        print(f"\n✅ LOADED TABLES from {TABLE_PATH}")
        print(f"   - Master Q-Table Size: {len(agent.q_table)}")
        print(f"   - Switch Q-Table Size: {len(agent.switch_table)}")
    else:
        print(f"\n⚠️ WARNING: Could not find {TABLE_PATH}.")

    print("\n" + "="*50)
    print(f"🤖 AGENT ONLINE:  {agent.username}")
    print("="*50)
    print(f"👉 Open your browser to: http://127.0.0.1:8000")
    print(f"👉 Search for user '{agent.username}'")
    print(f"👉 Challenge to: [Gen 1] Random Battle")
    print("="*50 + "\n")

    await agent.accept_challenges(None, 100)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Agent shutting down. Thanks for playing!")