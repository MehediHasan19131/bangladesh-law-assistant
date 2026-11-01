"""Retrieve -> prompt -> generate -> attach citations.

    python -m lawrag.rag "What does the constitution say about equality before law?"
"""

import argparse
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .index import Chunk, HybridIndex
from .llm import LLM, get_llm
from .text import is_bangla, to_ascii_digits

SYSTEM_PROMPT = """You answer questions about the laws of Bangladesh using ONLY the numbered sources provided.

Rules:
- Base every statement on the sources. Cite them inline with their numbers, like [1] or [2][3].
- Sources may be in Bangla or English; use them whatever language the question is in.
- Name the article or section you rely on (e.g. "Article 27", "ধারা ১০০").
- Be concise: a direct answer first, then the key conditions or exceptions.
- If the sources do not contain the answer, do not guess.
- This is general legal information, not legal advice."""

# The reply language and the exact "not found" sentence go next to the question, in one language only:
# small local models follow that much more reliably than a bilingual rule in the system prompt.
INSTRUCTIONS = {
    "en": ("Answer in English.", "I could not find this in the laws I have."),
    "bn": ("বাংলায় উত্তর দিন।", "প্রদত্ত আইনগুলোতে এ বিষয়ে তথ্য পাইনি।"),
}
NOT_FOUND_MARKERS = ("could not find", "couldn't find", "পাইনি", "পাওয়া যায়নি", "উল্লেখ নেই")
CITATION = re.compile(r"\[([0-9০-৯]{1,2})\]")  # models answering in Bangla may write [১]


@dataclass
class Citation:
    n: int
    chunk_id: str
    label: str
    title: str
    text: str
    url: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    sources: list[Citation]
    not_found: bool
    provider: str
    model: str
    retrieval_ms: int
    generation_ms: int
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def format_sources(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {c.label} — {c.title}\n{c.text}" for i, c in enumerate(chunks, 1))


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    answer_in, not_found = INSTRUCTIONS["bn" if is_bangla(question) else "en"]
    return (
        f"<sources>\n{format_sources(chunks)}\n</sources>\n\n"
        f"Question: {question}\n\n"
        f"{answer_in} Cite sources like [1]. If the sources do not answer the question, reply only: {not_found}"
    )


def cited_numbers(text: str, n_sources: int) -> list[int]:
    seen = []
    for m in CITATION.finditer(text):
        n = int(to_ascii_digits(m.group(1)))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def is_not_found(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in NOT_FOUND_MARKERS)


class Assistant:
    def __init__(self, index: HybridIndex, llm: LLM, k: int = 6, mode: str = "dense"):
        # dense beats hybrid on this bilingual corpus: BM25 cannot match English questions to Bangla
        # text, and fusing it in pushes correct results down (see results/retrieval.json)
        self.index, self.llm, self.k, self.mode = index, llm, k, mode
        index.embedder  # load the embedding model now so the first question isn't slow  # noqa: B018

    def ask(self, question: str, k: int | None = None, mode: str | None = None) -> Answer:
        start = time.perf_counter()
        chunks = [c for c, _ in self.index.search(question, k=k or self.k, mode=mode or self.mode)]
        retrieved = time.perf_counter()
        gen = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks))
        done = time.perf_counter()

        sources = [Citation(i, c.id, c.label, c.title, c.text, c.url) for i, c in enumerate(chunks, 1)]
        cited = [sources[n - 1] for n in cited_numbers(gen.text, len(sources))]
        return Answer(
            question=question,
            answer=gen.text,
            citations=cited,
            sources=sources,
            not_found=is_not_found(gen.text) and not cited,
            provider=self.llm.name,
            model=gen.model,
            retrieval_ms=round((retrieved - start) * 1000),
            generation_ms=round((done - retrieved) * 1000),
            usage={"input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    answer = Assistant(HybridIndex.load(args.index), get_llm(args.provider)).ask(args.question)
    print(answer.answer, "\n")
    for c in answer.citations:
        print(f"[{c.n}] {c.label} — {c.title}")
    print(f"\n{answer.provider}/{answer.model} · retrieval {answer.retrieval_ms} ms · "
          f"generation {answer.generation_ms} ms")


if __name__ == "__main__":
    main()
