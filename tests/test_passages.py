from atlandex_mcp.passages import ANCHOR_WORDS, MAX_PASSAGE_CHARS, question_terms, select_passage

from fakes import CHUNK_EVAL, CHUNK_INTRO


def test_question_terms_drop_function_words_and_short_tokens():
    assert question_terms("Why did dense retrieval lose to TF-IDF?") == {"dense", "retrieval", "lose", "idf"}
    assert question_terms("Was sagt er über die Suche?") == {"sagt", "suche"}


def test_picks_the_window_with_the_question_terms():
    passage = select_passage(CHUNK_EVAL, "Why did dense retrieval lose to TF-IDF?")
    assert "dense retrieval run lost to the plain TF-IDF baseline" in passage.text
    assert "coffee machine" not in passage.text
    assert {"dense", "retrieval", "idf"} <= set(passage.matched_terms)


def test_anchor_is_the_opening_words_of_the_passage():
    passage = select_passage(CHUNK_EVAL, "reciprocal rank fusion recall")
    words = passage.text.split()
    assert passage.anchor == " ".join(words[:ANCHOR_WORDS])
    assert passage.anchor in CHUNK_EVAL


def test_prefix_matching_finds_inflected_forms():
    passage = select_passage(CHUNK_EVAL, "how were embeddings ranked")
    assert "embedding model" in passage.text


def test_no_overlap_falls_back_to_the_opening_window():
    passage = select_passage(CHUNK_INTRO, "quantum chromodynamics lattice")
    assert passage.text.startswith("welcome back to the channel")
    assert passage.matched_terms == ()


def test_long_windows_are_cut_at_a_word_boundary():
    passage = select_passage(CHUNK_EVAL, "dense retrieval")
    assert len(passage.text) <= MAX_PASSAGE_CHARS + 2
    long_text = " ".join(["word"] * 400)
    cut = select_passage(long_text, "word", max_chars=100)
    assert cut.text.endswith(" …")
    assert len(cut.text) <= 102


def test_empty_chunk():
    passage = select_passage("   ", "anything")
    assert (passage.text, passage.anchor, passage.matched_terms) == ("", "", ())
