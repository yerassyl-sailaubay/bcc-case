from dataclasses import dataclass
from pathlib import Path
import re

import pymupdf
from markdown_it import MarkdownIt


@dataclass(frozen=True)
class Block:
    id: str
    location: str
    context: str
    text: str


def read_blocks(path: Path) -> list[Block]:
    blocks = []

    def add(location, context, text):
        if text.strip():
            blocks.append(Block(f"B{len(blocks) + 1:04}", location, context, text.strip()))

    if path.suffix.lower() == ".pdf":
        try:
            pdf = pymupdf.open(path)
        except Exception as exc:
            raise ValueError("Не удалось открыть PDF: файл повреждён или не является корректным документом PDF.") from exc
        with pdf:
            if pdf.needs_pass:
                raise ValueError("PDF защищён паролем.")
            if len(pdf) == 0:
                raise ValueError("PDF не содержит страниц.")
            for number, page in enumerate(pdf, 1):
                parts = page.get_text("blocks", sort=True)
                if not any(p[6] == 0 and p[4].strip() for p in parts):
                    raise ValueError(f"Страница {number} без текстового слоя. Проверьте PDF или примените OCR.")
                for p in parts:
                    if p[6] == 0:
                        add(f"стр. {number}", "", p[4])
    elif path.suffix.lower() in (".md", ".markdown"):
        text = path.read_text(encoding="utf-8-sig")
        lines = text.splitlines()
        headings = []
        header = ""
        in_thead = False
        for token in MarkdownIt("commonmark").enable("table").parse(text):
            if token.type == "thead_open":
                in_thead = True
                if token.map:
                    header = lines[token.map[0]]
                continue
            if token.type == "thead_close":
                in_thead = False
                continue
            if not token.map:
                continue
            start, end = token.map
            raw = "\n".join(lines[start:end])
            if token.type == "heading_open":
                level = int(token.tag[1:])
                headings = [(n, h) for n, h in headings if n < level]
                headings.append((level, raw.strip("# ")))
            if token.type == "tr_open" and in_thead:
                continue
            if token.type in {"paragraph_open", "tr_open", "fence", "code_block", "html_block"}:
                context = " / ".join(h for _, h in headings)
                if token.type == "tr_open" and header:
                    context += " | " + header
                add(f"строки {start + 1}–{end}", context, raw)
    else:
        raise ValueError("Поддерживаются Markdown и PDF с текстовым слоем.")
    if not blocks:
        raise ValueError("Документ не содержит текста требований.")
    if sum(len(b.text) + len(b.context) for b in blocks) > 150_000:
        raise ValueError("Документ превышает лимит полного анализа (150 000 символов с контекстом).")
    return blocks


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

