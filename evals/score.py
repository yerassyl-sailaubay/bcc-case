"""Summarize blinded engineering adjudication. Run after all ratings are saved."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("ratings", type=Path, help="JSON keyed by blind ID, with seed, false_blockers, supported_blockers, actionable_questions")
    args = parser.parse_args()
    ratings = json.loads(args.ratings.read_text())
    groups = defaultdict(Counter)
    stability = defaultdict(list)
    for path in sorted((args.run / "private").glob("*.json")):
        if path.name.endswith(".audit.json"):
            continue
        meta = json.loads(path.read_text())
        rating = ratings[meta["id"]]
        audit = json.loads(path.with_suffix(".audit.json").read_text())
        c = groups[meta["method"]]
        c["runs"] += 1
        for key in ("false_blockers", "supported_blockers", "actionable_questions"):
            c[key] += rating[key]
        if rating["seed"] is not None:
            c["seed_trials"] += 1
            c["seed_hits"] += int(rating["seed"])
            stability[meta["method"], meta["file"]].append(rating["seed"])
        else:
            c["clean_trials"] += 1
            c["clean_falsely_blocked"] += int(rating["false_blockers"] > 0)
        for f in audit["review"]["findings"]:
            c["verified_citations"] += len(f["evidence"])
            c["questions"] += 1
        c["seconds"] += audit["passport"]["seconds"]
        for call in audit["calls"]:
            c["input_tokens"] += call["input"]
            c["output_tokens"] += call["output"]
    for (method, _), trials in stability.items():
        if len(trials) == 3:
            groups[method]["seeds_found_all_three"] += int(all(trials))
    print(json.dumps(groups, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

