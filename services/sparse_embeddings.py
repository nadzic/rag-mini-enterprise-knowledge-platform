from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from qdrant_client.models import SparseVector

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_MAX_TERM_ID = (2**31) - 1


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def _term_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % _MAX_TERM_ID


def _to_sparse_vector(weights: dict[int, float]) -> SparseVector:
    if not weights:
        return SparseVector(indices=[], values=[])
    sorted_items = sorted(weights.items(), key=lambda item: item[0])
    return SparseVector(
        indices=[item[0] for item in sorted_items],
        values=[float(item[1]) for item in sorted_items],
    )


class BM25SparseEncoder:
    """BM25 sparse encoder for chunks and query text."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, k3: float = 7.0) -> None:
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self._idf: dict[int, float] = {}
        self._avg_doc_len: float = 0.0
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        tokenized_docs = [_tokenize(text) for text in texts]
        num_docs = len(tokenized_docs)
        if num_docs == 0:
            self._idf = {}
            self._avg_doc_len = 0.0
            self._fitted = True
            return

        doc_freq: Counter[int] = Counter()
        doc_lens = [len(tokens) for tokens in tokenized_docs]
        for tokens in tokenized_docs:
            for term in {_term_id(token) for token in tokens}:
                doc_freq[term] += 1

        self._avg_doc_len = sum(doc_lens) / num_docs if num_docs else 0.0
        self._idf = {
            term: math.log(1.0 + ((num_docs - df + 0.5) / (df + 0.5)))
            for term, df in doc_freq.items()
        }
        self._fitted = True

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        if not self._fitted:
            self.fit(texts)
        return [self._encode_document(_tokenize(text)) for text in texts]

    def encode_query(self, text: str) -> SparseVector:
        if not self._fitted:
            raise ValueError("BM25SparseEncoder must be fitted before encoding a query.")
        return self._encode_query(_tokenize(text))

    def _encode_document(self, tokens: list[str]) -> SparseVector:
        if not tokens:
            return SparseVector(indices=[], values=[])

        doc_len = len(tokens)
        tf = Counter(_term_id(token) for token in tokens)
        denom_norm = self.k1 * (1.0 - self.b + self.b * (doc_len / max(self._avg_doc_len, 1e-9)))

        weights: dict[int, float] = {}
        for term, term_tf in tf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            score = idf * ((term_tf * (self.k1 + 1.0)) / (term_tf + denom_norm))
            if score > 0.0:
                weights[term] = score
        return _to_sparse_vector(weights)

    def _encode_query(self, tokens: list[str]) -> SparseVector:
        if not tokens:
            return SparseVector(indices=[], values=[])

        qtf = Counter(_term_id(token) for token in tokens)
        weights: dict[int, float] = {}
        for term, term_qtf in qtf.items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            qtf_weight = ((self.k3 + 1.0) * term_qtf) / (self.k3 + term_qtf)
            score = idf * qtf_weight
            if score > 0.0:
                weights[term] = score
        return _to_sparse_vector(weights)


def build_bm25_encoder_and_sparse_chunks(
    chunks: list[str],
) -> tuple[BM25SparseEncoder, list[SparseVector]]:
    encoder = BM25SparseEncoder()
    sparse_chunks = encoder.encode_documents(chunks)
    return encoder, sparse_chunks
