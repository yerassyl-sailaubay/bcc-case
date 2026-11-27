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
    assert "не завершена" in out.read_text()
    assert "Готовность" not in out.read_text()
    assert "sensitive input" not in out.read_text()


def test_rejects_false_evidence():
    blocks = read_blocks(CASE)
    f = Finding(title="t",scope="set",kind="contradiction",severity="blocker",impact="both",
                criterion="c",evidence=[Citation(block=blocks[0].id,quote="fabricated")],
                scenario="s",outcomes=["x","y"],consequence="c",owner="o",question="q")
    with pytest.raises(ValueError, match="Цитата"):
        check_findings(Findings(findings=[f]), blocks)
    f.evidence[0].quote = blocks[0].text
    with pytest.raises(ValueError, match="двух"):
        check_findings(Findings(findings=[f]), blocks)


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

