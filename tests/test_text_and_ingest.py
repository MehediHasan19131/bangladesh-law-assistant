from lawrag.ingest import ACTS, decode, parse_sections
from lawrag.text import is_bangla, stem, to_ascii_digits, tokenize

PAGE = """
<div class="row lineremoves">
  <div class="col-sm-3 txt-head" style="">	Equality before law </div>
  <div class="col-sm-9 txt-details" id="sec-dec">	27. All citizens are equal before law.</div>
</div>
<div class="row lineremoves">
  <div class="col-sm-3 txt-head" style="">The state religion</div>
  <div class="col-sm-9 txt-details" id="sec-dec"><span class='footnote' title='Inserted by Act'>
    <span><sup class='en'><a href='1'>3</a></sup></span></span>[2A. The state religion of the Republic is Islam.]</div>
</div>
<div class="row lineremoves">
  <div class="col-sm-3 txt-head" style="">[বিলুপ্ত]</div>
  <div class="col-sm-9 txt-details" id="sec-dec">[***]</div>
</div>
"""

BANGLA_PAGE = """
<div class="row lineremoves">
  <div class="col-sm-3 txt-head">সংবিধান বাতিল</div>
  <div class="col-sm-9 txt-details">[৭ক। (১) কোন ব্যক্তি শক্তি প্রদর্শন করিলে</div>
</div>
"""


def test_parse_sections_numbers_footnotes_and_repeals():
    sections = parse_sections(PAGE, ACTS[0])
    assert [s["section"] for s in sections] == ["27", "2A"]  # repealed row dropped
    assert sections[0]["title"] == "Equality before law"
    assert sections[1]["text"].startswith("[2A. The state religion")  # footnote number "3" removed


def test_bangla_section_letters_match_english_ids():
    (section,) = parse_sections(BANGLA_PAGE, ACTS[1])
    assert section["section"] == "7A"


def test_decode_handles_utf16_and_utf8():
    assert decode("আইন".encode("utf-16")) == "আইন"
    assert decode("আইন".encode()) == "আইন"


def test_tokenize_mixed_language_and_digits():
    assert tokenize("Article ২৭ of the Constitution") == ["article", "27", "constitution"]


def test_stem_merges_inflections():
    assert stem("শ্রমিকের") == stem("শ্রমিকদের") == stem("শ্রমিককে") == "শ্রমিক"
    assert stem("ছুটি") == "ছুটি"  # too short to strip


def test_language_detection():
    assert is_bangla("শ্রমিকরা কত দিন ছুটি পান?")
    assert not is_bangla("How many days of leave?")
    assert to_ascii_digits("ধারা ১০০") == "ধারা 100"
