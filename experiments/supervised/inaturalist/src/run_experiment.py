from __future__ import annotations

import argparse
import json
import sys

from analyze import analyze
from evaluate import evaluate_all
from prepare import prepare
from train import dry_validate_migration, full, smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen iNaturalist supervised grouped-rotation experiment")
    parser.add_argument("stage", choices=["prepare", "smoke", "full", "evaluate", "analyze", "post-smoke", "all", "migration-dry-validate"])
    args = parser.parse_args()
    if args.stage == "migration-dry-validate":
        result = dry_validate_migration()
        print(f"SEED0_FILE_HASH: {result['seed0_file_hash']}")
        print(f"SEED1_FILE_HASH: {result['seed1_file_hash']}")
        print(f"SEED0_INITIALIZATION_MATCH: {result['seed0_initialization_match']}")
        print(f"SEED1_INITIALIZATION_MATCH: {result['seed1_initialization_match']}")
        print(f"M1_SEED0_STATUS: {result['m1_seed0_status']}")
        print(f"M2_SEED0_STATUS: {result['m2_seed0_status']}")
        print(f"M3_SEED0_RESUME_CHECKPOINT: {result['m3_seed0_resume_checkpoint']}")
        print(f"DRY_VALIDATION: {result['dry_validation']}")
        print("NEXT_RESUME_COMMAND: python src/run_experiment.py post-smoke")
        return
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("smoke", "all"):
        smoke()
    if args.stage in ("full", "post-smoke", "all"):
        full()
    if args.stage in ("evaluate", "post-smoke", "all"):
        rows = evaluate_all()
        print(json.dumps({"evaluation_rows": len(rows), "status": "COMPLETE"}, indent=2), flush=True)
    if args.stage in ("analyze", "post-smoke", "all"):
        analyze()


if __name__ == "__main__":
    main()
