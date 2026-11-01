"""Normalization and tokenization that work for both Bangla and English."""

import re
import unicodedata

BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
WORD = re.compile(r"[\u0980-\u09FF]+|[a-z0-9]+")
BANGLA_CHAR = re.compile(r"[\u0980-\u09FF]")

# Light suffix stripping so inflected Bangla forms meet in BM25: শ্রমিকের / শ্রমিকদের / শ্রমিককে -> শ্রমিক.
# Longest suffixes first; a stem must keep at least 3 code points (so ছুটি stays ছুটি).
BANGLA_SUFFIXES = sorted(
    ["গুলোর", "গুলো", "গুলি", "দিগকে", "দের", "দিগের", "য়ের", "েরা", "ের", "কে", "তে", "টির", "টি", "টা", "রা",
     "র", "য়", "ে"],
    key=len,
    reverse=True,
)
ENGLISH_STOPWORDS = set(
    "a an the of to in on for and or is are was be by with as at from that this it its can do does what who "
    "how when which i my me we our you your there their any under shall".split()
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def to_ascii_digits(text: str) -> str:
    return text.translate(BANGLA_DIGITS)


def stem(token: str) -> str:
    if not BANGLA_CHAR.match(token):
        return token
    for suffix in BANGLA_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    words = WORD.findall(to_ascii_digits(normalize(text)).lower())
    return [stem(w) for w in words if w not in ENGLISH_STOPWORDS]


def bangla_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if BANGLA_CHAR.match(c)) / len(letters) if letters else 0.0


def is_bangla(text: str) -> bool:
    return bangla_ratio(text) > 0.5
