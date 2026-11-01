import numpy as np
import pytest

from lawrag.evaluate import check_answer, first_gold_rank
from lawrag.index import rrf, split_text
from lawrag.rag import Assistant, cited_numbers, is_not_found

from .conftest import FakeLLM


def test_split_text_respects_limit_and_overlaps():
    text = " ".join(f"({i}) clause number {i} of the section." for i in range(200))
    parts = split_text(text, max_chars=300, overlap=50)
    assert len(parts) > 1
    assert all(len(p) <= 300 for p in parts)
    assert parts[1].split()[0] in parts[0]  # each window starts with the tail of the previous one


def test_short_text_is_one_chunk():
    assert split_text("short") == ["short"]


def test_rrf_rewards_agreement():
    bm25 = np.array([3.0, 2.0, 1.0])
    dense = np.array([0.1, 0.9, 0.8])
    fused = rrf([bm25, dense], pool=3)
    assert fused.argmax() == 1  # ranked 2nd and 1st beats 1st and 3rd


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_finds_the_right_section(index, mode):
    question = {"question": "within how many hours must an arrested person see a magistrate",
                "gold": ["constitution:33"]}
    assert first_gold_rank(index, question, mode) == 1


def test_bangla_query_finds_bangla_section(index):
    question = {"question": "শ্রমিক দিনে কত ঘণ্টা কাজ করবেন", "gold": ["labour:100"]}
    assert first_gold_rank(index, question, "bm25") == 1


def test_unknown_mode_rejected(index):
    with pytest.raises(ValueError):
        index.search("x", mode="magic")


def test_cited_numbers_dedupes_and_drops_out_of_range():
    assert cited_numbers("A [2] B [2][1] C [9]", n_sources=3) == [2, 1]


def test_cited_numbers_accepts_bangla_digits():
    assert cited_numbers("ধারা ১১৫ অনুযায়ী দশ দিন [১]। আরও [৩]", n_sources=6) == [1, 3]


def test_not_found_detection_both_languages():
    assert is_not_found("I could not find this in the laws I have.")
    assert is_not_found("প্রদত্ত আইনগুলোতে এ বিষয়ে তথ্য পাইনি।")
    assert not is_not_found("Article 27 says all citizens are equal [1].")


def test_assistant_attaches_citations(index):
    llm = FakeLLM("All citizens are equal before law under Article 27 [1].")
    answer = Assistant(index, llm, k=2).ask("Are citizens equal before the law?")
    assert [c.chunk_id for c in answer.citations] == [answer.sources[0].chunk_id]
    assert "<sources>" in llm.last_user and "Answer in English." in llm.last_user
    assert not answer.not_found


def test_check_answer_scores_grounding_and_language(index):
    llm = FakeLLM("Article 27: all citizens are equal [1].")
    answer = Assistant(index, llm, k=3).ask("Are all citizens equal before the law?")
    q = {"answerable": True, "facts": ["equal"], "gold": ["constitution:27"], "question": answer.question}
    checks = check_answer(q, answer)
    assert checks["fact"] and checks["cited"] and checks["language"]
    assert checks["grounded"] == (answer.citations[0].chunk_id.startswith("constitution-en:27"))


def test_check_answer_unanswerable(index):
    llm = FakeLLM("I could not find this in the laws I have.")
    answer = Assistant(index, llm).ask("What is the income tax rate?")
    assert check_answer({"answerable": False, "question": answer.question}, answer) == {"declined": True}
