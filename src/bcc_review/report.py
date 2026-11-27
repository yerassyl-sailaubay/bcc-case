from collections import Counter


def cell(value):
    return " ".join(value.split()).replace("|", "\\|")


def render(blocks, inventory, result, passport):
    sources = {b.id: b for b in blocks}
    rank = {"blocker": 0, "major": 1, "minor": 2}
    ordered = sorted(result.findings, key=lambda f: rank[f.severity])
    findings = [(f"D{i:03}", f) for i, f in enumerate(ordered, 1)]
    severity = {"blocker": "блокер", "major": "существенный", "minor": "замечание"}
    kind_label = {
        "contradiction": "противоречие",
        "omission": "пропуск",
        "ambiguity": "неоднозначность",
        "unverifiable": "непроверяемость",
        "other": "прочее",
    }
    unresolved = [e.block for e in inventory.entries if e.role == "unresolved"]
    template_count = sum(1 for e in inventory.entries if e.role == "template")
    claims = [(e.block, c) for e in inventory.entries for c in e.claims]
    lines = [f"# Проверка требований — {passport['document']}", "", "## Готовность", ""]
    for key, label in [("development", "Разработка"), ("testing", "Тестирование")]:
        relevant = [i for i, f in findings if f.impact in (key, "both") and f.severity == "blocker"]
        verdict = "требует решения " + ", ".join(relevant) if relevant else "подтверждённых блокеров не выявлено"
        if unresolved:
            verdict += ". Оценка частичная, требуется разбор " + ", ".join(unresolved)
        lines.append(f"- **{label}** — {verdict}.")
    lines.extend(["", "## Первые решения", ""])
    top_decisions = [item for item in findings if item[1].severity == "blocker"] or findings[:3]
    for i, f in top_decisions:
        lines += [f"**{i}. {f.title}** ({severity[f.severity]} · {kind_label[f.kind]})", "",
                  f"{f.scenario} {' / '.join(f.outcomes)}",
                  "", f"{f.consequence}", "", f"**{f.owner}** — {f.question}", ""]
        for c in f.evidence:
            lines.append(f"- {c.block}, {sources[c.block].location} — «{cell(c.quote)}»")
        lines.append("")
    if not findings:
        lines.append("Вопросов по подтверждённым нарушениям нет.")
    counts = Counter(f.severity for _, f in findings)
    lines += ["", "## Сводка и паспорт", "",
              f"Требований: {len(claims)} (проанализировано блоков: {len(blocks)}, шаблон/метаданные: {template_count}). "
              f"Нарушений: {len(findings)}. Блокеров: {counts['blocker']}. "
              f"Существенных: {counts['major']}. Замечаний: {counts['minor']}.",
              "", f"Критичные дефекты: {', '.join(i for i, f in findings if f.severity == 'blocker') or 'нет'}.",
              "", f"Дата: {passport['date']}. Модель: {passport['model']}. Версия: {passport['version']}.",
              f"SHA-256: `{passport['sha256']}`. Время: {passport['seconds']} с.",
              "", "## Реестр и проверка каждого требования", "",
              "| ID | ID в документе | Источник | Формулировка | Оценка | Связанные выводы |",
              "|---|---|---|---|---|---|"]
    for n, (bid, claim) in enumerate(claims, 1):
        block = sources[bid]
        links = ", ".join(i for i, f in findings if any(c.block == bid for c in f.evidence))
        lines.append("| " + " | ".join(map(cell, [f"R{n:03}", claim.original_id or "—",
                     f"{bid}, {block.location}", claim.statement, claim.assessment, links or "—"])) + " |")
    for scope, heading in [("individual", "Дефекты отдельных требований"), ("set", "Проверка всего набора")]:
        lines.extend(["", f"## {heading}", ""])
        if not any(f.scope == scope for _, f in findings):
            lines.append("Подтверждённых нарушений не выявлено.")
        for i, f in findings:
            if f.scope != scope:
                continue
            lines += [f"### {i}. {f.title}", "", f"**{severity[f.severity]} · {kind_label[f.kind]}** · {f.criterion}", "",
                      f.scenario, "", *[f"- {o}" for o in f.outcomes], "", f.consequence, "",
                      f"**Вопрос ({f.owner})** — {f.question}", ""]
            for c in f.evidence:
                b = sources[c.block]
                lines.append(f"- {b.id}, {b.location}, {cell(b.context)} — «{cell(c.quote)}»")
    lines.extend(["", "## Источники требований", ""])
    referenced = {bid for bid, _ in claims} | {c.block for _, f in findings for c in f.evidence}
    for b in blocks:
        if b.id in referenced:
            lines.append(f"- {b.id}, {b.location} — {cell(b.text)}")
    return "\n".join(lines) + "\n"

