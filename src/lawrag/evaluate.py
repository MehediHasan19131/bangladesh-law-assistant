"""Evaluate retrieval and end-to-end answers against eval/questions.jsonl.

    python -m lawrag.evaluate retrieval --index index --index index-e5   # compare embedders and search modes
    python -m lawrag.evaluate answers --limit 20                         # run the LLM on the questions

Retrieval metrics use the gold sections (language-independent, e.g. "constitution:33"):
  hit@k  share of answerable questions with a gold section in the top k
  MRR    mean reciprocal rank of the first gold section (top 10)

Answer checks are deterministic, so they cost nothing to re-run:
  fact      the answer contains an expected fact (e.g. "২৪" / "twenty four")
  cited     the answer cites at least one source
  grounded  at least one cited source is a gold section
  language  the answer is in the question's language
  declined  for unanswerable questions, the assistant says it could not find the answer
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .index import HybridIndex
from .llm import get_llm
from .rag import Assistant
from .text import is_bangla

MODES = ["bm25", "dense", "hybrid"]


def load_questions(path: Path = Path("eval/questions.jsonl")) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def first_gold_rank(index: HybridIndex, question: dict, mode: str, depth: int = 10) -> int | None:
    gold = set(question["gold"])
    for rank, (chunk, _) in enumerate(index.search(question["question"], k=depth, mode=mode), 1):
        if chunk.gold_key in gold:
            return rank
    return None


def retrieval_report(index: HybridIndex, questions: list[dict]) -> dict:
    answerable = [q for q in questions if q["answerable"]]
    report = {}
    for mode in MODES:
        ranks = [first_gold_rank(index, q, mode) for q in answerable]
        by_category = defaultdict(list)
        for q, r in zip(answerable, ranks, strict=True):
            by_category[q["category"]].append(r is not None and r <= 5)
        report[mode] = {
            "hit@1": mean(r == 1 for r in ranks),
            "hit@3": mean(r is not None and r <= 3 for r in ranks),
            "hit@5": mean(r is not None and r <= 5 for r in ranks),
            "mrr@10": mean(1 / r if r else 0 for r in ranks),
            "hit@5_by_category": {c: mean(v) for c, v in sorted(by_category.items())},
        }
    return report


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 0.0


def check_answer(question: dict, answer) -> dict:
    text = answer.answer.lower()
    cited_keys = {c.chunk_id.split("#")[0].replace("-en:", ":").replace("-bn:", ":") for c in answer.citations}
    if not question["answerable"]:
        return {"declined": answer.not_found}
    return {
        "fact": any(f.lower() in text for f in question["facts"]),
        "cited": bool(answer.citations),
        "grounded": bool(cited_keys & set(question["gold"])),
        "language": is_bangla(answer.answer) == is_bangla(question["question"]),
        "declined_wrongly": answer.not_found,
    }


def answers_report(assistant: Assistant, questions: list[dict], out: Path) -> dict:
    rows = []
    for i, q in enumerate(questions, 1):
        started = time.perf_counter()
        answer = assistant.ask(q["question"])
        checks = check_answer(q, answer)
        rows.append({**q, "answer": answer.answer, "cited": [c.chunk_id for c in answer.citations],
                     "checks": checks, "latency_ms": round((time.perf_counter() - started) * 1000)})
        print(f"[{i}/{len(questions)}] {q['id']} {checks}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    by_category = defaultdict(list)
    for r in answerable:
        by_category[r["category"]].append(r["checks"]["fact"])
    latencies = [r["latency_ms"] for r in rows]
    return {
        "provider": assistant.llm.name,
        "model": getattr(assistant.llm, "model", None),
        "questions": len(rows),
        "fact_accuracy": mean(r["checks"]["fact"] for r in answerable),
        "citation_rate": mean(r["checks"]["cited"] for r in answerable),
        "grounded_citation_rate": mean(r["checks"]["grounded"] for r in answerable),
        "language_match": mean(r["checks"]["language"] for r in answerable),
        "false_decline_rate": mean(r["checks"]["declined_wrongly"] for r in answerable),
        "correct_decline_rate": mean(r["checks"]["declined"] for r in unanswerable),
        "fact_accuracy_by_category": {c: mean(v) for c, v in sorted(by_category.items())},
        "latency_ms_p50": round(statistics.median(latencies)),
        "latency_ms_p90": round(sorted(latencies)[int(0.9 * (len(latencies) - 1))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["retrieval", "answers"])
    parser.add_argument("--index", type=Path, action="append", help="index dir(s); repeat to compare embedders")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_questions(args.questions)[: args.limit]
    indexes = args.index or [Path("index")]
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.what == "retrieval":
        report = {}
        for directory in indexes:
            index = HybridIndex.load(directory)
            report[index.embedder_name] = retrieval_report(index, questions)
            for mode, m in report[index.embedder_name].items():
                print(f"{index.embedder_name:40s} {mode:7s} hit@1={m['hit@1']:.3f} hit@5={m['hit@5']:.3f} "
                      f"mrr@10={m['mrr@10']:.3f}")
        (results_dir / "retrieval.json").write_text(json.dumps(report, indent=2))
    else:
        assistant = Assistant(HybridIndex.load(indexes[0]), get_llm(args.provider))
        name = f"answers_{assistant.llm.name}_{getattr(assistant.llm, 'model', '').replace(':', '-')}"
        report = answers_report(assistant, questions, results_dir / f"{name}.jsonl")
        (results_dir / f"{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
