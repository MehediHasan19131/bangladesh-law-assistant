import numpy as np
import pytest

from lawrag.index import Chunk, HybridIndex
from lawrag.llm import Generation

CHUNKS = [
    Chunk("constitution-en:27#0", "constitution", "en", "27", "Constitution, Article 27", "Equality before law",
          "27. All citizens are equal before law and are entitled to equal protection of law.", "http://x"),
    Chunk("constitution-en:33#0", "constitution", "en", "33", "Constitution, Article 33",
          "Safeguards as to arrest and detention",
          "33. Every person who is arrested shall be produced before the nearest magistrate within twenty four hours.",
          "http://x"),
    Chunk("labour-bn:100#0", "labour", "bn", "100", "বাংলাদেশ শ্রম আইন, ধারা 100", "দৈনিক কর্মঘণ্টা",
          "১০০। কোন প্রাপ্তবয়স্ক শ্রমিক কোন প্রতিষ্ঠানে সাধারণতঃ দৈনিক আট ঘণ্টার অধিক সময় কাজ করিবেন না।", "http://x"),
]


class FakeEmbedder:
    """Deterministic bag-of-words vectors, so dense search works without downloading a model."""

    vocab = ["equal", "law", "arrest", "magistrate", "hours", "শ্রমিক", "ঘণ্টা", "কাজ"]

    def encode(self, texts, **_):
        out = np.array([[t.lower().count(w) + 0.01 for w in self.vocab] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class FakeLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str = "Every citizen is equal before law [1]."):
        self.reply = reply

    def generate(self, system: str, user: str) -> Generation:
        self.last_user = user
        return Generation(self.reply, self.model, 10, 5)


@pytest.fixture
def index() -> HybridIndex:
    embedder = FakeEmbedder()
    idx = HybridIndex(CHUNKS, embedder.encode([f"{c.header}\n{c.text}" for c in CHUNKS]), "fake")
    idx._embedder = embedder
    return idx
