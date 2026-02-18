import os

ROOT_DIR = r"c:\Users\Robin\MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory"

# Walk and replace
count = 0
for root, dirs, files in os.walk(ROOT_DIR):
    if ".git" in root or ".venv" in root or "__pycache__" in root or "node_modules" in root:
        continue
    
    for file in files:
        if file.endswith(".py") or file.endswith(".md"): 
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Replace imports
                new_content = content.replace("from mnemocore.", "from mnemocore.")
                new_content = new_content.replace("import mnemocore.", "import mnemocore.")
                
                # Update references to src/core etc in comments/markdown if they look like paths?
                # For now stick to code imports.
                
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated {path}")
                    count += 1
            except Exception as e:
                print(f"Error reading/writing {path}: {e}")

print(f"Refactored {count} files.")
