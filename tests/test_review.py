import json
from pathlib import Path
import pymupdf
import pytest
from bcc_review import cli
from bcc_review.engine import analyze, check_findings, check_inventory
from bcc_review.report import render
from bcc_review.schema import Citation, Entry, Finding, Findings, Inventory
from bcc_review.source import read_blocks

CASE = Path("submission/requirements.md")


def test_real_source():
    blocks = read_blocks(CASE)
    row = next(b for b in blocks if "Февральский счётчик" in b.text)
    assert "Ожидаемый результат" in row.context
    assert "Критерии приемки" in row.context
    assert not any(b.text.strip().startswith("| Поле | Значение |") for b in blocks)
    assert not any(b.text.strip().startswith("| Исходные условия |") for b in blocks)
    example = next(b for b in blocks if '"result":"ERROR"' in b.text)
    assert "Successful" in example.context
    assert len(blocks) == len({b.id for b in blocks})
    for b in blocks:
        start, end = map(int, b.location.split()[1].split("–"))
        assert b.text in "\n".join(CASE.read_text().splitlines()[start-1:end])


def test_inventory_coverage():
    blocks = read_blocks(CASE)
    with pytest.raises(ValueError):
        check_inventory(Inventory(entries=[]), blocks)


def test_pdf_and_markdown(tmp_path):
    md = tmp_path / "input.md"
    md.write_text("The request expires after 48 hours.")
    pdf = tmp_path / "input.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((48, 48), md.read_text())
        doc.save(pdf)
    assert read_blocks(pdf)[0].text == read_blocks(md)[0].text
    assert read_blocks(pdf)[0].location == "стр. 1"


@pytest.mark.parametrize("mode", ["empty", "damaged", "scan", "missing_key", "api_error"])
def test_cli_failure(mode, monkeypatch, tmp_path):
    import openai
    path = tmp_path / ("case.pdf" if mode in ("damaged", "scan") else "case.md")
    out = tmp_path / "report.md"
    if mode == "scan":
        with pymupdf.open() as pdf:
            pdf.new_page()
            pdf.save(path)
    else:
        path.write_text("" if mode == "empty" else "Документ")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    if mode == "api_error":
        monkeypatch.setenv("OPENAI_API_KEY", "test-only")
        def interrupted(*args, **kwargs):
            raise openai.OpenAIError("sensitive input")
        monkeypatch.setattr(cli, "analyze", interrupted)
    monkeypatch.setattr("sys.argv", ["validator", "review", str(path), "--out", str(out)])
    assert cli.main() == 1
    body = out.read_text()
    assert "не завершена" in body
    assert "Готовность" not in body
    assert "sensitive input" not in body
    if mode == "damaged":
        assert "повреждён" in body
        assert "Failed to open" not in body
        assert str(tmp_path) not in body


def test_rejects_false_evidence():
    blocks = read_blocks(CASE)
    fabricated = Finding(title="t", scope="individual", kind="ambiguity", severity="major", impact="both",
                         criterion="c", evidence=[Citation(block=blocks[0].id, quote="fabricated quote")],
                         scenario="s", outcomes=["x", "y"], consequence="c", owner="o", question="q")
    with pytest.raises(ValueError, match="Цитата не найдена"):
        check_findings(Findings(findings=[fabricated]), blocks)
    assert fabricated.evidence[0].quote == "fabricated quote"

    single_contradiction = Finding(title="t", scope="set", kind="contradiction", severity="blocker", impact="both",
                                   criterion="c", evidence=[Citation(block=blocks[0].id, quote=blocks[0].text[:20])],
                                   scenario="s", outcomes=["x", "y"], consequence="c", owner="o", question="q")
    with pytest.raises(ValueError, match="двух"):
        check_findings(Findings(findings=[single_contradiction]), blocks)


def test_citation_preserves_literal_identifiers():
    blocks = read_blocks(CASE)
    card_block = next(b for b in blocks if "card_id" in b.text)
    bad = Finding(title="t", scope="individual", kind="ambiguity", severity="major", impact="both",
                  criterion="c", evidence=[Citation(block=card_block.id, quote="cardid")],
                  scenario="s", outcomes=["x", "y"], consequence="c", owner="o", question="q")
    with pytest.raises(ValueError, match="Цитата не найдена"):
        check_findings(Findings(findings=[bad]), blocks)

    ok = Finding(title="t", scope="individual", kind="ambiguity", severity="major", impact="both",
                 criterion="c", evidence=[Citation(block=card_block.id, quote="card_id")],
                 scenario="s", outcomes=["x", "y"], consequence="c", owner="o", question="q")
    check_findings(Findings(findings=[ok]), blocks)


def test_readiness_surfaces_major_risks_when_no_blockers():
    blocks = read_blocks(Path("evals/development/transfers-a.md"))
    inv = Inventory(entries=[Entry(block=b.id, role="template", claims=[]) for b in blocks])
    only_major = Findings(findings=[
        Finding(title="Major omission", scope="individual", kind="omission", severity="major", impact="both",
                criterion="c", evidence=[Citation(block=blocks[0].id, quote=blocks[0].text[:12])],
                scenario="s", outcomes=["a", "b"], consequence="c", owner="o", question="Q?")
    ])
    passport = dict(document="doc.md", date="2026-11-27", model="m", version="v", sha256="0", seconds=1.0)
    rendered = render(blocks, inv, only_major, passport)
    assert "есть существенные риски D001" in rendered


def test_inventory_strips_synthetic_block_ids_and_template_claims():
    blocks = read_blocks(Path("evals/development/transfers-a.md"))
    inv = Inventory(entries=[
        Entry(block=blocks[0].id, role="requirement", claims=[
            dict(original_id="B0001", statement="s1", assessment="a1"),
            dict(original_id="Ф-1", statement="s2", assessment="a2"),
        ]),
        *[Entry(block=b.id, role="template", claims=[
            dict(original_id="", statement="ignored template", assessment="meta")
        ]) for b in blocks[1:]],
    ])
    check_inventory(inv, blocks)
    assert inv.entries[0].claims[0].original_id == ""
    assert inv.entries[0].claims[1].original_id == "Ф-1"
    assert all(e.claims == [] for e in inv.entries[1:])


def test_render_shows_all_blockers_in_top_decisions():
    blocks = read_blocks(Path("evals/development/transfers-a.md"))
    inv = Inventory(entries=[Entry(block=b.id, role="template", claims=[]) for b in blocks])
    five_blockers = Findings(findings=[
        Finding(title=f"Blocker {i}", scope="set", kind="contradiction", severity="blocker", impact="both",
                criterion="c", evidence=[Citation(block=blocks[0].id, quote=blocks[0].text[:12])],
                scenario="s", outcomes=["a", "b"], consequence="c", owner="o", question=f"Q{i}?")
        for i in range(1, 6)
    ])
    passport = dict(document="doc.md", date="2026-11-27", model="m", version="v", sha256="0", seconds=1.0)
    rendered = render(blocks, inv, five_blockers, passport)
    top_section = rendered.split("## Первые решения", 1)[1].split("## Сводка и паспорт", 1)[0]
    for code in ("D001", "D002", "D003", "D004", "D005"):
        assert code in top_section


def test_fixtures():
    labels = json.loads(Path("evals/labels.json").read_text())
    assert len(labels) == 12
    for left, right in zip(labels[::2], labels[1::2]):
        assert not left["seed"] and right["seed"]
        assert Path(left["file"]).read_text() != Path(right["file"]).read_text()
        assert read_blocks(Path(right["file"]))


def test_pipeline_without_network(monkeypatch, tmp_path):
    class Stub:
        calls = [{"model":"test"}]
        def ask(self, schema, instruction, data, check):
            if schema is Inventory:
                result = Inventory(entries=[Entry(block=b["id"],role="requirement", claims=[
                    dict(original_id="", statement=b["text"], assessment="Проверяемо.")
                ]) for b in data])
            else:
                result = Findings(findings=[])
            check(result)
            return result
    monkeypatch.setattr(cli, "Reviewer", Stub)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    path = Path("evals/development/transfers-a.md")
    out, audit = tmp_path / "report.md", tmp_path / "run.json"
    cli.run(path, out, audit)
    assert "Реестр" in out.read_text() and "R001" in out.read_text()
    assert json.loads(audit.read_text())["passport"]["model"] == "test"



def test_missing_inventory_block_is_visible():
    blocks = read_blocks(Path("evals/development/transfers-a.md"))
    inv = Inventory(entries=[Entry(block=blocks[0].id, role="template", claims=[])])
    check_inventory(inv, blocks)
    assert [e.block for e in inv.entries] == [b.id for b in blocks]
    assert [e.role for e in inv.entries[1:]] == ["unresolved"] * (len(blocks) - 1)


def test_input_cannot_be_used_as_report(tmp_path, monkeypatch):
    path = tmp_path / "case.md"
    path.write_text("Исходное требование")
    monkeypatch.setattr("sys.argv", ["validator", "review", str(path), "--out", str(path)])
    with pytest.raises(SystemExit):
        cli.main()
    assert path.read_text() == "Исходное требование"


def test_inventory_accepts_address_with_location():
    blocks = read_blocks(Path("evals/development/transfers-a.md"))
    inv = Inventory(entries=[Entry(block=f"{b.id} — {b.location}", role="evidence", claims=[]) for b in blocks])
    check_inventory(inv, blocks)
    assert all(e.role == "evidence" for e in inv.entries)
    assert [e.block for e in inv.entries] == [b.id for b in blocks]
