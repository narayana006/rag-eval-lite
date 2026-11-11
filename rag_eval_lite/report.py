"""Plain-text summary, run comparison and a single-file HTML report."""

from __future__ import annotations

from html import escape


def format_summary(result: dict) -> str:
    lines = [f"questions: {result['n_questions']}   k: {result['k']}", ""]
    lines.append(f"{'metric':<16}{'mean':>8}{'n':>6}")
    for name, agg in result["summary"].items():
        lines.append(f"{name:<16}{agg['mean']:>8.3f}{agg['n']:>6}")
    misses = [row for row in result["questions"] if row["scores"].get(f"hit@{result['k']}") == 0.0]
    if misses:
        lines += ["", f"no relevant passage in top {result['k']}:"]
        lines += [f"  {row['id']}  {row['question']}" for row in misses]
    return "\n".join(lines)


def format_comparison(before: dict, after: dict) -> str:
    """Side-by-side means for two saved results, with the change."""
    names = list(before["summary"])
    names += [name for name in after["summary"] if name not in before["summary"]]
    lines = [f"{'metric':<16}{'before':>8}{'after':>8}{'change':>9}"]
    for name in names:
        old = before["summary"].get(name, {}).get("mean")
        new = after["summary"].get(name, {}).get("mean")
        old_s = f"{old:.3f}" if old is not None else "-"
        new_s = f"{new:.3f}" if new is not None else "-"
        delta = f"{new - old:+.3f}" if old is not None and new is not None else "-"
        lines.append(f"{name:<16}{old_s:>8}{new_s:>8}{delta:>9}")
    return "\n".join(lines)


_STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1c1c1c}
table{border-collapse:collapse;width:100%;margin:1rem 0}
th,td{border:1px solid #d6d6d6;padding:.35rem .55rem;text-align:left;vertical-align:top}
th{background:#f2f2f2}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.bad{background:#fde8e8}.ok{background:#e9f7ec}
.hit{font-weight:600;color:#17692d}.muted{color:#777}
"""


def _cell(value) -> str:
    if value is None:
        return '<td class="num muted">-</td>'
    css = "ok" if value >= 0.999 else ("bad" if value <= 0.001 else "")
    return f'<td class="num {css}">{value:.2f}</td>'


def render_html(result: dict, title: str = "rag-eval-lite report") -> str:
    names = list(result["summary"])
    out = [
        "<!doctype html>",
        f'<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>',
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>{escape(title)}</h1>",
        f"<p>{result['n_questions']} questions, k = {result['k']}</p>",
        "<h2>Summary</h2><table><tr><th>Metric</th><th>Mean</th><th>Questions scored</th></tr>",
    ]
    for name in names:
        agg = result["summary"][name]
        out.append(f"<tr><td>{escape(name)}</td><td class=\"num\">{agg['mean']:.3f}</td><td class=\"num\">{agg['n']}</td></tr>")
    out.append("</table><h2>Per question</h2><table><tr><th>Id</th><th>Question</th><th>Retrieved (relevant in bold)</th><th>Answer</th>")
    out += [f"<th>{escape(name)}</th>" for name in names]
    out.append("</tr>")
    for row in result["questions"]:
        relevant = set(row["relevant_ids"])
        retrieved = "<br>".join(
            f'<span class="hit">{escape(i)}</span>' if i in relevant else escape(i) for i in row["retrieved_ids"]
        ) or '<span class="muted">nothing</span>'
        answer = escape(row["answer"]) if row["answer"] is not None else '<span class="muted">-</span>'
        if row.get("expected_answer"):
            answer += f'<br><span class="muted">expected: {escape(row["expected_answer"])}</span>'
        out.append(f"<tr><td>{escape(row['id'])}</td><td>{escape(row['question'])}</td><td>{retrieved}</td><td>{answer}</td>")
        out += [_cell(row["scores"].get(name)) for name in names]
        out.append("</tr>")
    out.append("</table></body></html>")
    return "\n".join(out) + "\n"
