"""Run from the repository root. Existing completed trials are reusable after interruption."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from dotenv import load_dotenv

from bcc_review.cli import run, write_atomic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def organizer_variants(folder):
    source = Path("submission/requirements.md")
    text = source.read_text()
    old = "Февральский счётчик вырос"
    if text.count(old) != 1:
        raise ValueError("Изменился исходный пример приёмки.")
    corrected = folder / "organizer-corrected.md"
    corrected.write_text(text.replace(old, "Январский счётчик вырос"))
    head, part = text.split("## Часть A.", 1)
    a, b = part.split("## Часть B.", 1)
    reordered = folder / "organizer-reordered.md"
    reordered.write_text((head + "## Часть B." + b + "\n\n## Часть A." + a).replace("Ф-", "Пункт-"))
    return [source, corrected, reordered]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["development", "holdout", "organizer", "arithmetic"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(Path.cwd() / ".env")
    folder = args.out
    folder.mkdir(exist_ok=True, parents=True)
    (folder / "private").mkdir(exist_ok=True)
    (folder / "blind").mkdir(exist_ok=True)
    paths = organizer_variants(folder) if args.split == "organizer" else sorted(Path("evals", args.split).glob("*.md"))
    trials = 3 if args.split == "holdout" else 1
    jobs = [(p, method, i) for p in paths for method in ("agent", "baseline") for i in range(trials)]
    random.SystemRandom().shuffle(jobs)
    def execute(job):
        path, method, rep = job
        name = f"{path.stem}-{method}-{rep+1}"
        private = folder / "private"
        record = private / (name + ".json")
        if record.exists():
            return
        report = private / (name + ".md")
        audit = private / (name + ".audit.json")
        run(path, report, audit, method == "baseline")
        data = json.loads(audit.read_text())
        blind_id = uuid4().hex[:10]
        # Remove run metadata and filename, which would reveal the treatment.
        public = folder / "blind" / (blind_id + ".json")
        write_atomic(public, json.dumps({"source":path.read_text(), "review":data["review"]},
                                         ensure_ascii=False, indent=2))
        write_atomic(record, json.dumps(dict(id=blind_id, file=str(path), sha256=digest(path),
                                             method=method, repeat=rep+1), ensure_ascii=False))
        print(f"Готово: {name}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(execute, job): job for job in jobs}
        failures = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                path, method, trial = futures[future]
                failures.append(dict(file=str(path), method=method, trial=trial + 1, error=type(exc).__name__))
        write_atomic(folder / "failures.json", json.dumps(failures, indent=2))
        if failures:
            raise SystemExit(f"Незавершённых испытаний: {len(failures)}")


if __name__ == "__main__":
    main()

