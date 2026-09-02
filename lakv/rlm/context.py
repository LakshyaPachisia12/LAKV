"""External context environment (spec sections 6-8).

Phase 1 scope: the context lives as plain text in memory (no lazy
file-backed streaming yet — that's real work for contexts too large to hold
as a Python string, deferred until a benchmark actually needs it). What IS
implemented now is the token-aware addressing spec section 7 asks for,
because that's the piece LAKV research actually needs: every chunk gets a
stable id plus token/char boundaries and a depth/parent-chunk lineage, so a
later pass can map "RLM accessed chunk X" directly onto "these token
positions entered the KV cache."

Token boundaries use the real tokenizer's offset mapping when one is passed
in (accurate). Without a tokenizer (e.g. in unit tests with the mock
backend), token position falls back to a whitespace-word approximation —
this is clearly non-authoritative and is labeled as such in Chunk.metadata.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lakv.rlm.errors import InvalidChunkError


@dataclass
class Chunk:
    chunk_id: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    depth: int
    parent_chunk_id: Optional[str]
    text: str
    approximate_tokens: bool = False  # True if no real tokenizer was available


class Context:
    """A read-only view over one piece of text, with chunking/search/slice
    operations. One Context per RLM run (or per recursive sub-context)."""

    def __init__(self, text: str, tokenizer=None, source_id: str = "root"):
        self.text = text
        self.tokenizer = tokenizer
        self.source_id = source_id
        self._chunks: Dict[str, Chunk] = {}
        self._next_chunk_seq = 0

        if tokenizer is not None:
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            self._token_ids: List[int] = enc["input_ids"]
            self._offsets: List[tuple] = enc["offset_mapping"]
        else:
            # whitespace-word approximation — one "token" per word, offsets
            # from re.finditer so char positions are still real.
            self._token_ids = None
            self._offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    @property
    def length(self) -> int:
        """Token count (real if tokenizer given, else word-count approximation)."""
        return len(self._offsets)

    def _char_range_for_token_range(self, token_start: int, token_end: int) -> tuple:
        token_end = min(token_end, len(self._offsets))
        if token_start >= token_end:
            return (0, 0)
        char_start = self._offsets[token_start][0]
        char_end = self._offsets[token_end - 1][1]
        return char_start, char_end

    def _new_chunk_id(self) -> str:
        cid = f"chunk_{self._next_chunk_seq:06d}"
        self._next_chunk_seq += 1
        return cid

    def slice(self, token_start: int, token_end: int) -> str:
        char_start, char_end = self._char_range_for_token_range(token_start, token_end)
        return self.text[char_start:char_end]

    def chunk(
        self,
        chunk_size: int,
        overlap: int = 0,
        parent_chunk_id: Optional[str] = None,
        depth: int = 0,
    ) -> List[Chunk]:
        """Fixed-size (optionally overlapping) token chunking — spec section 8's
        baseline strategy. Registers each chunk so get_chunk() can resolve it."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        step = max(1, chunk_size - overlap)
        chunks = []
        start = 0
        n = self.length
        if n == 0:
            return chunks
        while start < n:
            end = min(start + chunk_size, n)
            char_start, char_end = self._char_range_for_token_range(start, end)
            c = Chunk(
                chunk_id=self._new_chunk_id(),
                token_start=start,
                token_end=end,
                char_start=char_start,
                char_end=char_end,
                depth=depth,
                parent_chunk_id=parent_chunk_id,
                text=self.text[char_start:char_end],
                approximate_tokens=(self.tokenizer is None),
            )
            self._chunks[c.chunk_id] = c
            chunks.append(c)
            if end >= n:
                break
            start += step
        return chunks

    def chunk_by_paragraphs(
        self, paragraphs: List[str], parent_chunk_id: Optional[str] = None, depth: int = 0
    ) -> List[Chunk]:
        """Register externally-provided paragraph boundaries (e.g. HotpotQA's
        already-split distractor paragraphs) as chunks, instead of re-deriving
        boundaries from self.text. Char/token offsets are computed against
        self.text via a straightforward search, so this only works correctly
        when `paragraphs` are verbatim substrings of the context text."""
        chunks = []
        cursor = 0
        for p in paragraphs:
            idx = self.text.find(p, cursor)
            if idx == -1:
                idx = self.text.find(p)
            if idx == -1:
                # paragraph text doesn't appear verbatim (e.g. it was built by
                # joining sentences with different whitespace) — fall back to
                # a synthetic chunk with no reliable char offset in this context.
                char_start, char_end = 0, len(p)
            else:
                char_start, char_end = idx, idx + len(p)
                cursor = char_end
            token_start = self._token_index_for_char(char_start)
            token_end = self._token_index_for_char(char_end)
            c = Chunk(
                chunk_id=self._new_chunk_id(),
                token_start=token_start,
                token_end=token_end,
                char_start=char_start,
                char_end=char_end,
                depth=depth,
                parent_chunk_id=parent_chunk_id,
                text=p,
                approximate_tokens=(self.tokenizer is None),
            )
            self._chunks[c.chunk_id] = c
            chunks.append(c)
        return chunks

    def _token_index_for_char(self, char_pos: int) -> int:
        for i, (s, e) in enumerate(self._offsets):
            if s >= char_pos:
                return i
        return len(self._offsets)

    def search(self, pattern: str, case_sensitive: bool = False) -> List[Chunk]:
        """Return already-registered chunks whose text matches `pattern`
        (plain substring or regex). Only searches chunks created so far via
        chunk()/chunk_by_paragraphs() — search() does not implicitly chunk
        the whole context first, so it stays cheap on large contexts."""
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), flags)
        return [c for c in self._chunks.values() if rx.search(c.text)]

    def get_chunk(self, chunk_id: str) -> Chunk:
        if chunk_id not in self._chunks:
            raise InvalidChunkError(f"unknown chunk_id: {chunk_id!r}")
        return self._chunks[chunk_id]

    def list_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())

    def metadata(self) -> dict:
        return {
            "source_id": self.source_id,
            "length_tokens": self.length,
            "length_chars": len(self.text),
            "n_chunks_registered": len(self._chunks),
            "token_addressing_approximate": self.tokenizer is None,
        }
