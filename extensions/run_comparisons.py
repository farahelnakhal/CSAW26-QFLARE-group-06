"""Run the QFLARE timing and round-budget bonus studies.

From the repository root: python extensions/run_comparisons.py
The root attack.py and released evaluator files are never edited.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from challenge import prepare, train  # noqa: E402
from score import RULES, compute_score, release_hashes  # noqa: E402
from scheduled_attack import SCHEDULES, attack_for  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402


def load_clean():
    path = ROOT / "outputs" / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["rules"] != RULES or result["release_sha256"] != release_hashes():
        raise ValueError("The required result does not match the released benchmark")
    return {(r["seed"], r["defense"]): r for r in result["runs"]}


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=SCHEDULES)
    parser.add_argument("--rerun", action="store_true", help="regenerate the selected results")
    args = parser.parse_args()
    clean_results = load_clean()
    names = [args.setting] if args.setting else list(SCHEDULES)
    source_hash = sha256((ROOT / "extensions" / "scheduled_attack.py").read_bytes()).hexdigest()
    if args.rerun:
        for name in names:
            category = "timing" if name.startswith("timing_") else "budget"
            (ROOT / "extensions" / category / f"{name}.json").unlink(missing_ok=True)

    with threadpool_limits(limits=1):
        for seed in RULES["public_seeds"]:
            data = prepare(seed)
            for defense in RULES["defenses"]:
                clean = clean_results[(seed, defense)]
                if clean["test_ids"] != data["test_ids"] or clean["y_true"] != data["labels"][data["test"]].tolist():
                    raise ValueError("Cached clean run uses a different test split")
                for name in names:
                    category = "timing" if name.startswith("timing_") else "budget"
                    path = ROOT / "extensions" / category / f"{name}.json"
                    if path.exists():
                        result = json.loads(path.read_text(encoding="utf-8"))
                        if result["attack_source_sha256"] != source_hash or result["release_sha256"] != release_hashes():
                            raise ValueError(f"Existing result is stale: {path}")
                    else:
                        result = {
                            "study": category,
                            "setting": name,
                            "scheduled_rounds": list(SCHEDULES[name]),
                            "method": "full-radius max-stall, opposite the honest pair mean",
                            "clean_source": "outputs/result.json",
                            "attack_source_sha256": source_hash,
                            "release_sha256": release_hashes(),
                            "rules": RULES,
                            "runs": [],
                        }
                    if any(r["seed"] == seed and r["defense"] == defense for r in result["runs"]):
                        continue
                    print(f"{name}: seed {seed}, {defense}", flush=True)
                    attacked = train(data, defense, attack_for(SCHEDULES[name]))
                    if attacked["attack_rounds"] != list(SCHEDULES[name]):
                        raise ValueError(f"Applied rounds differ from schedule for {name}")
                    result["runs"].append({
                        "seed": seed,
                        "defense": defense,
                        "test_ids": data["test_ids"],
                        "y_true": clean["y_true"],
                        "clean_predictions": clean["clean_predictions"],
                        "attacked_predictions": attacked["predictions"],
                        "attack_rounds": attacked["attack_rounds"],
                    })
                    if len(result["runs"]) == len(RULES["public_seeds"]) * len(RULES["defenses"]):
                        result["summary"] = compute_score(result["runs"])
                    save(path, result)
    print("Completed selected comparisons", flush=True)


if __name__ == "__main__":
    main()
