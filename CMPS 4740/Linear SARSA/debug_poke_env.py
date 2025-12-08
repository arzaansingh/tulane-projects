import os
import sys
import poke_env

print(f"poke-env location: {poke_env.__file__}")
package_dir = os.path.dirname(poke_env.__file__)

print("\n--- Searching for Core Files ---")
targets = ['pokemon.py', 'move.py', 'battle.py']

found_any = False

for root, dirs, files in os.walk(package_dir):
    for filename in files:
        if filename in targets:
            found_any = True
            rel_path = os.path.relpath(root, package_dir)
            
            # Construct the likely import path
            import_path = "poke_env"
            if rel_path != ".":
                import_path += "." + rel_path.replace(os.sep, ".")
            
            print(f"\n✅ Found {filename}!")
            print(f"   Folder: {root}")
            print(f"   Likely Import: from {import_path} import {filename.capitalize().replace('.py', '')}")

if not found_any:
    print("\n❌ CRITICAL: The files are missing entirely. Your installation might be corrupted.")
            
print("\n--- Top Level Folders in poke_env ---")
print([d for d in os.listdir(package_dir) if os.path.isdir(os.path.join(package_dir, d))])