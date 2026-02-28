import sys
from huggingface_hub import HfApi

def upload_to_hf():
    api = HfApi()
    
    # We are uploading the whole repo but excluding internal/unwanted files
    # User explicitly requested to NOT include the newsletter
    exclude_patterns = [
        ".git/**",
        ".venv/**",
        ".pytest_cache/**",
        "__pycache__/**",
        "data/**",
        "demo_error*.log",
        "*newsletter*",
        "*nyhetsbrev*",
        ".gemini/**",
        ".claude/**",
        ".tmp_phase43_tests/**"
    ]
    
    print("Uploading to HF...")
    api.upload_folder(
        folder_path="c:/Users/Robin/MnemoCore-Infrastructure-for-Persistent-Cognitive-Memory",
        repo_id="Granis87/MnemoCore",
        repo_type="model",
        ignore_patterns=exclude_patterns,
        commit_message="Release 4.5.0: Added demo script, fixed HDV JSON bugs, applied HF YAML tags"
    )
    print("Upload completed successfully!")

if __name__ == "__main__":
    upload_to_hf()
