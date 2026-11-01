"""Download laws from bdlaws.minlaw.gov.bd (official Laws of Bangladesh site) and split them into sections.

    python -m lawrag.ingest            # writes data/sections.jsonl
"""

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from .text import normalize, to_ascii_digits

BASE = "http://bdlaws.minlaw.gov.bd"


@dataclass(frozen=True)
class Act:
    key: str  # groups language versions of the same law, used for evaluation gold labels
    act_id: int
    lang: str
    name: str
    name_en: str
    unit: str  # what a section is called: Article / অনুচ্ছেদ / ধারা


ACTS = [
    Act("constitution", 367, "en", "The Constitution of the People's Republic of Bangladesh",
        "Constitution of Bangladesh", "Article"),
    Act("constitution", 957, "bn", "গণপ্রজাতন্ত্রী বাংলাদেশের সংবিধান", "Constitution of Bangladesh (Bangla)", "অনুচ্ছেদ"),
    Act("labour", 952, "bn", "বাংলাদেশ শ্রম আইন, ২০০৬", "Bangladesh Labour Act, 2006", "ধারা"),
    Act("consumer", 1014, "bn", "ভোক্তা অধিকার সংরক্ষণ আইন, ২০০৯", "Consumer Rights Protection Act, 2009", "ধারা"),
    Act("rti", 1011, "bn", "তথ্য অধিকার আইন, ২০০৯", "Right to Information Act, 2009", "ধারা"),
]

ROW = re.compile(
    r'class="col-sm-3 txt-head"[^>]*>(?P<title>.*?)</div>\s*'
    r'<div class="col-sm-9 txt-details"[^>]*>(?P<body>.*?)</div>\s*</div>',
    re.S,
)
FOOTNOTE_MARK = re.compile(r"<sup[^>]*>.*?</sup>", re.S)
TAG = re.compile(r"<[^>]+>")
SECTION_NO = re.compile(r"^[\W_]{0,6}?([0-9০-৯]+)\s*([A-Za-zকখগঘঙ]?)\s*[.।৷]")
BANGLA_SUFFIX_LETTER = str.maketrans("কখগঘঙ", "ABCDE")  # ৭ক == 7A, so both language versions share ids
REPEALED = re.compile(r"^\[?\s*(বিলুপ্ত|Omitted|Repealed)", re.I)


def decode(raw: bytes) -> str:
    """The site serves some pages as UTF-16 with a BOM and others as UTF-8."""
    if raw[:2] in (b"\xfe\xff", b"\xff\xfe"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


def html_to_text(fragment: str) -> str:
    fragment = FOOTNOTE_MARK.sub("", fragment)
    fragment = re.sub(r'<div class="(clbr|na)">.*?</div>', "\n", fragment)
    return normalize(html.unescape(TAG.sub(" ", fragment))).lstrip("'\"> ")


def parse_sections(page: str, act: Act) -> list[dict]:
    sections = []
    for i, match in enumerate(ROW.finditer(page)):
        title, text = html_to_text(match["title"]), html_to_text(match["body"])
        if not text or REPEALED.match(title) or text.strip("[]* ") == "":
            continue
        number = SECTION_NO.match(text)
        if number:
            section = to_ascii_digits(number.group(1)) + number.group(2).upper().translate(BANGLA_SUFFIX_LETTER)
        else:
            section = f"x{i}"
        sections.append({
            "act": act.key,
            "act_id": act.act_id,
            "act_name": act.name,
            "act_name_en": act.name_en,
            "lang": act.lang,
            "unit": act.unit,
            "section": section,
            "title": title,
            "text": text,
            "url": f"{BASE}/act-details-{act.act_id}.html",
        })
    return sections


def fetch(act: Act) -> str:
    resp = requests.get(f"{BASE}/act-details-{act.act_id}.html", timeout=120)
    resp.raise_for_status()
    return decode(resp.content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/sections.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for act in ACTS:
            sections = parse_sections(fetch(act), act)
            for s in sections:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"{act.name_en:45s} {len(sections):4d} sections")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

