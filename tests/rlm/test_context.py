import pytest

from lakv.rlm.context import Context
from lakv.rlm.errors import InvalidChunkError


def test_length_word_approximation_without_tokenizer():
    ctx = Context("one two three four five")
    assert ctx.length == 5


def test_fixed_chunking_no_overlap():
    ctx = Context(" ".join(f"w{i}" for i in range(10)))
    chunks = ctx.chunk(chunk_size=4)
    assert len(chunks) == 3  # 4, 4, 2
    assert chunks[0].token_start == 0 and chunks[0].token_end == 4
    assert chunks[1].token_start == 4 and chunks[1].token_end == 8
    assert chunks[2].token_start == 8 and chunks[2].token_end == 10
    # chunks tile the text with no gaps or overlaps
    assert chunks[0].char_end <= chunks[1].char_start


def test_fixed_chunking_with_overlap():
    ctx = Context(" ".join(f"w{i}" for i in range(10)))
    chunks = ctx.chunk(chunk_size=4, overlap=2)
    assert chunks[0].token_start == 0 and chunks[0].token_end == 4
    assert chunks[1].token_start == 2 and chunks[1].token_end == 6


def test_chunk_ids_are_unique_and_resolvable():
    ctx = Context(" ".join(f"w{i}" for i in range(10)))
    chunks = ctx.chunk(chunk_size=3)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert ctx.get_chunk(c.chunk_id) is c


def test_get_chunk_invalid_id_raises():
    ctx = Context("some text here")
    with pytest.raises(InvalidChunkError):
        ctx.get_chunk("chunk_999999")


def test_search_only_matches_registered_chunks():
    ctx = Context("alpha beta gamma delta epsilon zeta eta theta")
    # nothing registered yet — search finds nothing even though "beta" is in the text
    assert ctx.search("beta") == []
    ctx.chunk(chunk_size=2)
    matches = ctx.search("beta")
    assert len(matches) == 1
    assert "beta" in matches[0].text


def test_slice_matches_token_range():
    ctx = Context("alpha beta gamma delta")
    assert ctx.slice(0, 2).strip() == "alpha beta"
    assert ctx.slice(2, 4).strip() == "gamma delta"


def test_chunk_by_paragraphs_verbatim_substrings():
    p1 = "First paragraph text."
    p2 = "Second paragraph text."
    text = f"{p1}\n\n{p2}"
    ctx = Context(text)
    chunks = ctx.chunk_by_paragraphs([p1, p2])
    assert len(chunks) == 2
    assert chunks[0].text == p1
    assert chunks[1].text == p2
    assert chunks[0].char_start < chunks[1].char_start


def test_metadata_reports_approximate_when_no_tokenizer():
    ctx = Context("some words here")
    meta = ctx.metadata()
    assert meta["token_addressing_approximate"] is True
    assert meta["length_tokens"] == 3


def test_empty_context_chunking():
    ctx = Context("")
    assert ctx.length == 0
    assert ctx.chunk(chunk_size=10) == []
