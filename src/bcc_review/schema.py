from typing import Literal
from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(Record):
    original_id: str
    statement: str
    assessment: str


class Entry(Record):
    block: str
    role: Literal["requirement", "evidence", "template", "unresolved"]
    claims: list[Claim]


class Inventory(Record):
    entries: list[Entry]


class Citation(Record):
    block: str
    quote: str


class Finding(Record):
    title: str
    scope: Literal["individual", "set"]
    kind: Literal["contradiction", "omission", "ambiguity", "unverifiable", "other"]
    severity: Literal["blocker", "major", "minor"]
    impact: Literal["development", "testing", "both"]
    criterion: str
    evidence: list[Citation]
    scenario: str
    outcomes: list[str]
    consequence: str
    owner: str
    question: str


class Findings(Record):
    findings: list[Finding]


class Baseline(Record):
    inventory: Inventory
    review: Findings

