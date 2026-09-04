#!/usr/bin/env python3
"""Safely inspect or remove city folders that are not in cities_split.json."""

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Inspect or remove orphan city directories.")
    parser.add_argument("--run_dir", default="ckpts", help="directory containing cities_split.json")
    parser.add_argument("--data_dir", default="data", help="directory containing numeric city folders")
    parser.add_argument("--apply", action="store_true", help="actually delete orphan folders")
    return parser.parse_args()


def load_legal_city_ids(run_dir):
    split_file = Path(run_dir) / "cities_split.json"
    with split_file.open(encoding="utf-8") as f:
        payload = json.load(f)

    legal_ids = set(payload["cities"])
    legal_ids.update(payload["seen_cities"])
    legal_ids.update(payload["unseen_cities"])
    return {str(city_id) for city_id in legal_ids}


def find_orphan_city_dirs(data_dir, legal_ids):
    data_path = Path(data_dir)
    return sorted(
        sub_dir
        for sub_dir in data_path.iterdir()
        if sub_dir.is_dir() and sub_dir.name.isdigit() and sub_dir.name not in legal_ids
    )


def main():
    args = parse_args()
    legal_ids = load_legal_city_ids(args.run_dir)
    orphan_dirs = find_orphan_city_dirs(args.data_dir, legal_ids)

    if not orphan_dirs:
        print("No orphan city directories found.")
        return

    action = "Removing" if args.apply else "Would remove"
    for orphan_dir in orphan_dirs:
        print(f"{action}: {orphan_dir}")
        if args.apply:
            shutil.rmtree(orphan_dir)

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete these directories.")


if __name__ == "__main__":
    main()
