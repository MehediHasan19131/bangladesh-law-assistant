"""Chunking + hybrid retrieval (BM25 keyword search fused with multilingual dense embeddings).

    python -m lawrag.index                                   # build index/ with the default embedder
    python -m lawrag.index --embedder intfloat/multilingual-e5-base
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .text import tokenize

DEFAULT_EMBEDDER = "BAAI/bge-m3"
MAX_CHARS = 1500
OVERLAP = 200
RRF_K = 60

# e5 models were trained with these prefixes; others take raw text
PREFIXES = {"e5": ("query: ", "passage: ")}


@dataclass
class Chunk:
    id: str  # constitution-en:27#0
    act: str
    lang: str
    section: str
    label: str  # human-readable citation, e.g. "Constitution of Bangladesh, Article 27"
    title: str
    text: str
    url: str

    @property
    def gold_key(self) -> str:
        """Language-independent id used by the evaluation set: constitution:27."""
        return f"{self.act}:{self.section}"

    @property
    def header(self) -> str:
        return f"{self.label} — {self.title}"


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split long sections at sentence/clause boundaries into overlapping windows."""
    if len(text) <= max_chars:
        return [text]
    pieces = re.split(r"(?<=[।৷.;:])\s+|\s+(?=\([0-9০-৯ক-হa-z]{1,3}\)\s)", text)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = current[-overlap:].split(" ", 1)[-1]  # carry the tail over, starting at a word boundary
        current = f"{current} {piece}".strip()
        while len(current) > max_chars:  # a single piece longer than the window
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap:]
    if current:
        chunks.append(current)
    return chunks


def make_chunks(sections: list[dict]) -> list[Chunk]:
    chunks = []
    for s in sections:
        number = s["section"]
        label = f"{s['act_name']}, {s['unit']} {number}"
        for i, part in enumerate(split_text(s["text"])):
            chunks.append(Chunk(
                id=f"{s['act']}-{s['lang']}:{number}#{i}",
                act=s["act"], lang=s["lang"], section=number, label=label,
                title=s["title"], text=part, url=s["url"],
            ))
    return chunks


def load_sections(path: Path = Path("data/sections.jsonl")) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None, embedder_name: str | None):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder_name = embedder_name
        self._embedder = None
        self.bm25 = BM25Okapi([tokenize(f"{c.title} {c.text}") for c in chunks])

    # ---- building / persistence -------------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], embedder_name: str = DEFAULT_EMBEDDER) -> "HybridIndex":
        index = cls(chunks, None, embedder_name)
        _, passage_prefix = index._prefixes()
        index.embeddings = index.embedder.encode(
            [f"{passage_prefix}{c.header}\n{c.text}" for c in chunks],
            batch_size=16, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        return index

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "meta.json").write_text(json.dumps({"embedder": self.embedder_name, "chunks": len(self.chunks)}))

    @classmethod
    def load(cls, directory: Path) -> "HybridIndex":
        chunks = [Chunk(**json.loads(line)) for line in (directory / "chunks.jsonl").read_text("utf-8").splitlines()]
        meta = json.loads((directory / "meta.json").read_text())
        return cls(chunks, np.load(directory / "embeddings.npy"), meta["embedder"])

    # ---- search -----------------------------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def _prefixes(self) -> tuple[str, str]:
        return next((p for key, p in PREFIXES.items() if key in (self.embedder_name or "")), ("", ""))

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(tokenize(query))

    def dense_scores(self, query: str) -> np.ndarray:
        query_prefix, _ = self._prefixes()
        q = self.embedder.encode([f"{query_prefix}{query}"], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.embeddings @ q

    def search(self, query: str, k: int = 6, mode: str = "hybrid", pool: int = 50) -> list[tuple[Chunk, float]]:
        if mode == "bm25":
            scores = self.bm25_scores(query)
        elif mode == "dense":
            scores = self.dense_scores(query)
        elif mode == "hybrid":
            scores = rrf([self.bm25_scores(query), self.dense_scores(query)], pool)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def rrf(score_lists: list[np.ndarray], pool: int, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion: sum of 1/(k + rank) over each ranking's top `pool` results."""
    fused = np.zeros_like(score_lists[0], dtype=float)
    for scores in score_lists:
        for rank, i in enumerate(np.argsort(-scores)[:pool]):
            fused[i] += 1.0 / (k + rank + 1)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", type=Path, default=Path("data/sections.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("index"))
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    args = parser.parse_args()

    chunks = make_chunks(load_sections(args.sections))
    print(f"{len(chunks)} chunks from {args.sections}")
    HybridIndex.build(chunks, args.embedder).save(args.out)
    print(f"-> {args.out}/ ({args.embedder})")


if __name__ == "__main__":
    main()
