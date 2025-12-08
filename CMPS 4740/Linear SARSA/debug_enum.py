import os
import poke_env

print(f"poke-env location: {poke_env.__file__}")
package_dir = os.path.dirname(poke_env.__file__)

print("\n--- Searching for 'class Status' and 'class Effect' ---")

targets = ["class Status", "class Effect"]
found_files = []

for root, dirs, files in os.walk(package_dir):
    for filename in files:
        if filename.endswith(".py"):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for target in targets:
                        if target in content:
                            rel_path = os.path.relpath(filepath, os.path.dirname(package_dir))
                            print(f"✅ Found '{target}' in: {rel_path}")
                            found_files.append(rel_path)
            except Exception:
                pass

if not found_files:
    print("❌ Could not find definitions in source files.")
else:
    print("\n--- Suggested Imports ---")
    for f in found_files:
        module_path = f.replace(os.sep, ".").replace(".py", "")
        print(f"Likely import: from {module_path} import ...")

print("\n--- Checking poke_env.data contents ---")
try:
    import poke_env.data
    # Check if they are exposed in data
    if hasattr(poke_env.data, 'Status'):
        print("✅ poke_env.data.Status exists")
    else:
        print("❌ poke_env.data.Status MISSING")
        
    if hasattr(poke_env.data, 'Effect'):
        print("✅ poke_env.data.Effect exists")
    else:
        print("❌ poke_env.data.Effect MISSING")
except ImportError:
    print("Could not import poke_env.data")