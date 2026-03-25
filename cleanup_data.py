"""
One-time cleanup script: removes orphaned .tmp state files from data/
Run once before your demo: python cleanup_data.py
"""
import os
import glob

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def cleanup():
    patterns = [
        os.path.join(DATA_DIR, "*.tmp"),
        os.path.join(DATA_DIR, "*.tmp.json"),
        os.path.join(DATA_DIR, "state_live_*.json"),       # old timestamped .json files
        os.path.join(DATA_DIR, "state_live_*.json.tmp"),   # old timestamped .tmp files
        os.path.join(DATA_DIR, "state_live_*.tmp"),        # any other .tmp variants
    ]
    total = 0
    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            try:
                os.remove(f)
                total += 1
            except Exception as e:
                print(f"  Could not delete {f}: {e}")
    print(f"Cleanup done. Removed {total} orphaned files from data/")

if __name__ == "__main__":
    cleanup()
