from pathlib import Path

import pytest

from bcc_review.arithmetic import calculations
from bcc_review.source import read_blocks


@pytest.mark.parametrize(("path", "equal"), [
    ("holdout/refunds-a", True), ("holdout/refunds-b", False),
    ("arithmetic/duration-a", True), ("arithmetic/duration-b", False),
])
def test_rate_matches_exact_table_values(path, equal):
    blocks = read_blocks(Path(f"evals/{path}.md"))
    check, = calculations(blocks)
    assert check["equal"] is equal
    assert check["rule"] == blocks[0].id
    assert check["row"] == blocks[-1].id


def test_missing_equivalence_produces_no_comparison():
    assert calculations(read_blocks(Path("evals/holdout/parcel-b.md"))) == []


def test_inflected_units_and_inline_currency_cells(tmp_path):
    doc = tmp_path / "case.md"
    doc.write_text(
        "Для расчёта 1 тенге = 100 тиынов.\n\n"
        "| Сумма, тиын | Итог |\n"
        "|---|---|\n"
        "| 12 345 | 123,45 ₸ |\n"
    )
    check, = calculations(read_blocks(doc))
    assert check["equal"] is True
    assert check["expected_decimal"] == "123.45"
