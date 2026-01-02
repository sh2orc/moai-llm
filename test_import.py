import os
import sys
rank = int(os.environ.get("RANK", 0))
print(f"[TEST] Rank {rank}: Script started!", flush=True)
sys.stdout.flush()
