"""Check explicit unit equalities against numeric columns; interpretation stays with the reviewer."""
from fractions import Fraction
from itertools import permutations
import re

NUMBER = r"[-+]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
UNIT = r"(?:[^\W\d_]+|[₸€$₽])"
RATE = re.compile(rf"(?<![\w.,])({NUMBER})\s*({UNIT})\s*=\s*({NUMBER})\s*({UNIT})(?!\w)")
ALIASES = {"₸": "тенге", "kzt": "тенге", "€": "евро"}


def unit_key(text):
    return ALIASES.get(text.casefold(), text.casefold())


def number(text):
    return Fraction("".join(text.split()).replace(",", "."))


def calculations(blocks):
    rates = []
    for block in blocks:
        for match in RATE.finditer(block.text):
            left, src, right, dst = match.groups()
            if number(left) > 0 and number(right) > 0 and unit_key(src) != unit_key(dst):
                rates.append((block.id, match.group(), unit_key(src), unit_key(dst), number(right) / number(left)))

    if not rates:
        return []

    comparisons = []
    for row in blocks:
        if " | |" not in row.context or "|" not in row.text:
            continue
        header = row.context.rsplit(" | |", 1)[1].strip().strip("|").split("|")
        cells = row.text.strip().strip("|").split("|")
        if len(header) != len(cells):
            continue
        columns = [(h.strip(), {unit_key(u) for u in re.findall(UNIT, h)}, number(v.strip()))
                   for h, v in zip(header, cells) if re.fullmatch(NUMBER, v.strip())]
        for first, second in permutations(columns, 2):
            for rule, quote, src, dst, factor in rates:
                if src not in first[1] or dst not in second[1]:
                    continue
                expected = first[2] * factor
                comparisons.append(dict(rule=rule, equality=quote, row=row.id,
                                        input_column=first[0], output_column=second[0],
                                        input=str(first[2]), expected_exact=str(expected),
                                        stated_exact=str(second[2]), equal=expected == second[2]))
                if len(comparisons) >= 100:
                    return comparisons
    return comparisons
