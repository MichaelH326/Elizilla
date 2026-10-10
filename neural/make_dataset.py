#!/usr/bin/env python3
"""
Build the neural Elizilla's training data end to end (DATA_PLAN.md):

    profiles -> simulated conversations -> Elizilla + teacher replies -> clean -> Mira JSONL

    python3 neural/make_dataset.py --teacher none --out-dir data/neural        # templates only, minutes
    python3 neural/make_dataset.py --teacher fake --out-dir /tmp/neural         # pipeline test
    python3 neural/make_dataset.py --teacher models/teacher.gguf --out-dir data/neural

In CI the slow teacher steps run as parallel shards (profiles.py and build.py
take --shard/--shards); this script runs them in one process.
Outputs in --out-dir: neural_data.jsonl, neural_data_val.jsonl, dataset_card.json,
plus the intermediate files.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="none")
    p.add_argument("--per-situation", type=int, default=2)
    p.add_argument("--per-profile", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-unique", type=float, default=15.0)
    args = p.parse_args()
    d = args.out_dir
    os.makedirs(d, exist_ok=True)
    f = lambda name: os.path.join(d, name)

    run(os.path.join(HERE, "profiles.py"), "--teacher", args.teacher, "--per-situation",
        str(args.per_situation), "--seed", str(args.seed), "--out", f("profiles.jsonl"))
    run(os.path.join(HERE, "simulate.py"), f("profiles.jsonl"), "--per-profile", str(args.per_profile),
        "--seed", str(args.seed), "--out", f("people.jsonl"))
    run(os.path.join(HERE, "build.py"), f("people.jsonl"), "--teacher", args.teacher,
        "--seed", str(args.seed), "--out", f("built.jsonl"))
    run(os.path.join(HERE, "clean.py"), f("built.jsonl"), "--out", f("clean.jsonl"),
        "--card", f("dataset_card.json"), "--seed", str(args.seed), "--min-unique", str(args.min_unique))
    run(os.path.join(HERE, "export.py"), f("clean.jsonl"), "--out", f("neural_data.jsonl"),
        "--val-out", f("neural_data_val.jsonl"))

    with open(f("dataset_card.json")) as fh:
        card = json.load(fh)
    card["teacher"] = args.teacher if args.teacher in ("none", "fake") else {
        "file": os.path.basename(args.teacher), "sha256": sha256(args.teacher)}
    card["seed"] = args.seed
    card["inputs_sha256"] = {name: sha256(os.path.join(ROOT, name)) for name in (
        "neural/situations.txt", "neural/profiles.py", "neural/simulate.py", "neural/build.py",
        "neural/checks.py", "neural/mirror.py", "neural/teacher.py", "lexicon.py", "rules.py",
        "corpus.txt", "neo_eliza.py", "slm.py", "sentence.py")}
    with open(f("dataset_card.json"), "w") as fh:
        json.dump(card, fh, indent=1)
    print(f"done: {f('neural_data.jsonl')} (card: {f('dataset_card.json')})")


if __name__ == "__main__":
    main()
