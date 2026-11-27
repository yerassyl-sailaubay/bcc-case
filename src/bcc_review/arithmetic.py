"""Check explicit unit equalities against numeric columns; interpretation stays with the reviewer."""
from decimal import Decimal
from fractions import Fraction
from itertools import combinations
import re

NUMBER = r"[-+]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
UNIT = r"(?:[^\W\d_]+|[₸€$₽])"
RATE = re.compile(rf"(?<![\w.,])({NUMBER})\s*({UNIT})\s*=\s*({NUMBER})\s*({UNIT})(?!\w)")
CELL_VALUE = re.compile(rf"({NUMBER})(?:\s*({UNIT}))?")
ALIASES = {
    "₸": "тенге", "kzt": "тенге", "тенге": "тенге",
    "тиын": "тиын", "тиына": "тиын", "тиынов": "тиын", "тиынах": "тиын", "тиыны": "тиын", "тийын": "тиын", "tiyn": "тиын",
    "€": "евро", "eur": "евро", "евро": "евро",
    "$": "доллар", "usd": "доллар", "доллар": "доллар", "доллара": "доллар", "долларов": "доллар",
    "₽": "рубль", "rub": "рубль", "руб": "рубль", "рубль": "рубль", "рубля": "рубль", "рублей": "рубль",
    "с": "сек", "сек": "сек", "секунда": "сек", "секунды": "сек", "секунд": "сек", "секундах": "сек",
    "мин": "мин", "минута": "мин", "минуты": "мин", "минут": "мин", "минутах": "мин",
    "ч": "час", "час": "час", "часа": "час", "часов": "час", "часах": "час",
    "д": "день", "дн": "день", "день": "день", "дня": "день", "дней": "день", "сут": "день", "сутки": "день", "суток": "день",
    "г": "г", "гр": "г", "грамм": "г", "грамма": "г", "граммов": "г",
    "кг": "кг", "килограмм": "кг", "килограмма": "кг", "килограммов": "кг",
    "м": "м", "метр": "м", "метра": "м", "метров": "м",
    "км": "км", "километр": "км", "километра": "км", "километров": "км",
}


def unit_key(text):
    return ALIASES.get(text.casefold(), text.casefold())


def number(text):
    return Fraction("".join(text.split()).replace(",", "."))


def format_decimal(value: Fraction) -> str:
    denom = value.denominator
    while denom % 2 == 0:
        denom //= 2
    while denom % 5 == 0:
        denom //= 5
    if denom == 1:
        dec = Decimal(value.numerator) / Decimal(value.denominator)
        return format(dec.normalize(), "f")
    return f"{float(value):.6g}"


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
        columns = []
        for h, v in zip(header, cells):
            m = CELL_VALUE.fullmatch(v.strip())
            if not m:
                continue
            units = {unit_key(u) for u in re.findall(UNIT, h)}
            if m.group(2):
                units.add(unit_key(m.group(2)))
            columns.append((h.strip(), units, number(m.group(1))))
        for first, second in combinations(columns, 2):
            for rule, quote, src, dst, factor in rates:
                if src in first[1] and dst in second[1]:
                    applied = factor
                elif dst in first[1] and src in second[1]:
                    applied = Fraction(1, 1) / factor
                else:
                    continue
                expected = first[2] * applied
                comparisons.append(dict(rule=rule, equality=quote, row=row.id,
                                        input_column=first[0], output_column=second[0],
                                        input=str(first[2]), expected_exact=str(expected),
                                        expected_decimal=format_decimal(expected),
                                        stated_exact=str(second[2]), equal=expected == second[2]))
                if len(comparisons) >= 100:
                    return comparisons
    return comparisons
