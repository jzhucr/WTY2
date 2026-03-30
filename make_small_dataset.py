#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def month_key(ts: str) -> str:
    # expected format: YYYY-MM-DD HH:MM:SS
    if isinstance(ts, str) and len(ts) >= 7:
        return ts[:7]
    return "unknown"


def stratified_sample(rows: List[Dict], max_rows: int, seed: int) -> List[Dict]:
    if len(rows) <= max_rows:
        return rows

    rng = random.Random(seed)
    by_month: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_month[month_key(str(r.get("timestamp", "")))].append(r)

    months = sorted(by_month.keys())
    total = len(rows)

    selected: List[Dict] = []
    leftovers: List[Dict] = []

    for m in months:
        group = by_month[m]
        # proportionally allocate slots; at least 1 per month if possible
        quota = max(1, round(max_rows * (len(group) / total)))
        if quota >= len(group):
            selected.extend(group)
        else:
            picked = rng.sample(group, quota)
            selected.extend(picked)
            picked_ids = {x.get("message_id") for x in picked}
            for item in group:
                if item.get("message_id") not in picked_ids:
                    leftovers.append(item)

    # fix size exactly to max_rows
    if len(selected) > max_rows:
        selected = rng.sample(selected, max_rows)
    elif len(selected) < max_rows:
        need = max_rows - len(selected)
        if len(leftovers) > need:
            selected.extend(rng.sample(leftovers, need))
        else:
            selected.extend(leftovers)

    selected.sort(key=lambda r: (r.get("timestamp_unix") is None, r.get("timestamp_unix", 0), r.get("message_id", "")))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a smaller clean_messages subset for GitHub.")
    parser.add_argument("--input", required=True, help="Path to clean_messages.jsonl")
    parser.add_argument("--output", required=True, help="Path to output small jsonl")
    parser.add_argument("--max-rows", type=int, default=12000, help="Maximum rows for output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)
    sampled = stratified_sample(rows, max_rows=args.max_rows, seed=args.seed)
    write_jsonl(output_path, sampled)

    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(sampled)}")
    print(f"output_file={output_path}")


if __name__ == "__main__":
    main()
