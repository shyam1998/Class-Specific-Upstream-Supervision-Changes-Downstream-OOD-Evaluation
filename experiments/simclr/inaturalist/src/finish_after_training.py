"""Infrastructure-only handoff: run frozen downstream stages after training exits."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import MODELS, ROOT, SEEDS, now


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m src.finish_after_training TRAINING_PID")
    pid = int(sys.argv[1])
    finals = [ROOT / "checkpoints" / f"{model}_seed{seed}" / "epoch_200.pt"
              for seed in SEEDS for model in MODELS]
    while alive(pid):
        time.sleep(30)
    if not all(path.is_file() for path in finals):
        payload = {"status": "TRAINING_EXITED_INCOMPLETE", "timestamp_utc": now(),
                   "training_pid": pid, "completed_final_checkpoints": sum(path.is_file() for path in finals),
                   "missing": [str(path) for path in finals if not path.is_file()]}
        (ROOT / "logs" / "post_pipeline_failure.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise SystemExit(1)
    with (ROOT / "logs" / "post_training_console.log").open("a", encoding="utf-8") as handle:
        for stage in ("evaluate", "analyze"):
            subprocess.run([sys.executable, "-u", "-m", "src.run_experiment", stage],
                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
