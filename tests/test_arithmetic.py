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
