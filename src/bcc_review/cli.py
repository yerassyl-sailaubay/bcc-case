import argparse
import hashlib
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import monotonic

import openai
from dotenv import load_dotenv

from .engine import Reviewer, analyze
from .report import render
from .source import read_blocks


def agent_version():
    root = Path(__file__).resolve().parent
    rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True)
    return rev.stdout.strip() if rev.returncode == 0 else version("bcc-review")


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        tmp.write(content)
    os.replace(tmp.name, path)


def run(path, out, audit=None, baseline=False):
    if len({p.resolve() for p in [path, out, *([audit] if audit else [])]}) != (3 if audit else 2):
        raise ValueError("Пути входного документа и выходных файлов должны различаться.")
    started = monotonic()
    blocks = read_blocks(path)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Задайте OPENAI_API_KEY в окружении или в .env.")
    reviewer = Reviewer()
    inventory, result = analyze(blocks, reviewer, baseline)
    passport = dict(document=path.name, date=datetime.now().astimezone().isoformat(timespec="seconds"),
                    model=", ".join(dict.fromkeys(c["model"] for c in reviewer.calls)), version=agent_version(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seconds=round(monotonic()-started, 2))
    if audit:
        data = dict(passport=passport, calls=reviewer.calls, inventory=inventory.model_dump(),
                    review=result.model_dump())
        write_atomic(audit, json.dumps(data, ensure_ascii=False, indent=2))
    write_atomic(out, render(blocks, inventory, result, passport))


def main():
    parser = argparse.ArgumentParser(description="Проверка качества требований")
    review = parser.add_subparsers(dest="command", required=True).add_parser("review")
    review.add_argument("input", type=Path)
    review.add_argument("--out", type=Path, required=True)
    review.add_argument("--audit", type=Path)
    review.add_argument("--baseline", action="store_true", help="Однопроходный контроль для оценки")
    args = parser.parse_args()
    paths = [args.input, args.out, *([args.audit] if args.audit else [])]
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error("Вход и выходы должны иметь разные пути.")
    load_dotenv(Path.cwd() / ".env")
    try:
        run(args.input, args.out, args.audit, args.baseline)
    except Exception as exc:
        if isinstance(exc, openai.OpenAIError):
            detail = f"Сбой OpenAI API ({type(exc).__name__}). Проверьте ключ, квоту и соединение."
        else:
            detail = str(exc)
        message = "# Проверка не завершена\n\n" + detail + "\n"
        write_atomic(args.out, message)
        print(message, file=sys.stderr)
        return 1
    return 0

