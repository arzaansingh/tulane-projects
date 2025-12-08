import subprocess
import sys
import time

#Battles per session = 5,000
TOTAL_SESSIONS = 10
SCRIPT_NAME = "poke_train_updated.py" #File name of bot

print(f"Starting Training. Goal: {TOTAL_SESSIONS} sessions.")

for i in range(1, TOTAL_SESSIONS + 1):
    print(f"\n==================================================")
    print(f"   SESSION {i}/{TOTAL_SESSIONS} STARTING")
    print(f"==================================================")
    
    # Run the training script and wait for it to finish
    try:
        subprocess.run([sys.executable, SCRIPT_NAME], check=True)
    except subprocess.CalledProcessError:
        print(f"Session {i} crashed! Stopping sequence.")
        break
    except KeyboardInterrupt:
        print("\nManual stop detected. Exiting.")
        break
        
    print(f"Session {i} complete. Memory cleared.")
    time.sleep(2) # Short pause to ensure file handles are closed

print("\n All training sessions finished!")