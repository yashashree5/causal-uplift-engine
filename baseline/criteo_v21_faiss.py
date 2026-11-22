"""Reuse identical float32 queries while retaining stock one-dimensional FAISS search."""

import sys
import time

import faiss
import numpy as np


class _NeighborRows:
    """Expand stock neighbor labels only for rows requested by the estimator."""

    def __init__(self, labels, inverse):
        self._labels = labels
        self._inverse = inverse

    def __getitem__(self, rows):
        return self._labels[self._inverse[rows]]


class _ReusedFlatL2:
    def __init__(self, backend, chunk_size=2048, log=True):
        self._backend = backend
        self._index = backend.IndexFlatL2(1)
        self._chunk_size = chunk_size
        self._log = log

    def add(self, values):
        return self._index.add(values)

    def search(self, queries, k):
        query_values = np.asarray(queries, dtype=np.float32).reshape(-1)
        unique, inverse = np.unique(query_values, return_inverse=True)
        threshold = int(self._backend.cvar.distance_compute_blas_threshold)
        uses_blas = len(query_values) >= threshold
        label_dtype = np.int32 if self._index.ntotal - 1 <= np.iinfo(np.int32).max else np.int64
        labels = np.empty((len(unique), k), dtype=label_dtype)
        blocks = (len(unique) + self._chunk_size - 1) // self._chunk_size
        started = time.perf_counter()
        if self._log:
            print(
                f"FAISS retrieval: {self._index.ntotal:,} arm references, {len(query_values):,} queries, "
                f"{len(unique):,} distinct, {blocks:,} blocks, {self._backend.omp_get_max_threads()} threads",
                flush=True,
            )
        for block, start in enumerate(range(0, len(unique), self._chunk_size), start=1):
            values = unique[start : start + self._chunk_size]
            submitted = values
            if uses_blas and len(values) < threshold:
                # Stock switches to its matrix-product distance kernel at this
                # threshold. Repeated padding preserves that numerical path.
                submitted = np.pad(values, (0, threshold - len(values)), mode="edge")
            distances, found = self._index.search(submitted[:, None], k)
            labels[start : start + len(values)] = found[: len(values)]
            del distances, found
            if self._log and (block == 1 or block % 10 == 0 or block == blocks):
                print(
                    f"FAISS retrieval: {block:,}/{blocks:,} blocks in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        # The pinned estimator discards distances and gathers label rows in
        # batches. Keep a compressed label matrix and no distance matrix.
        return None, _NeighborRows(labels, inverse)


class _FaissProxy:
    def __init__(self, backend):
        self._backend = backend

    def IndexFlatL2(self, dimension):
        if dimension == 1:
            return _ReusedFlatL2(self._backend)
        return self._backend.IndexFlatL2(dimension)

    def __getattr__(self, name):
        return getattr(self._backend, name)


def install(estimator_class):
    """Patch only the loaded pinned estimator module's FAISS reference."""
    module = sys.modules[estimator_class.__module__]
    if not isinstance(module.faiss, _FaissProxy):
        module.faiss = _FaissProxy(module.faiss)


def verify():
    """Check stock equivalence for small, repeated, tied, and chunked queries."""
    rng = np.random.default_rng(82718)
    threshold = int(faiss.cvar.distance_compute_blas_threshold)
    k = 1024
    cases = (
        ("direct with signed zeros", np.tile([0, -0.0, -0.004, 0.004], 3), True),
        ("BLAS threshold with repeated values", np.resize([0, -0.004, 0.004], threshold), True),
        ("BLAS with ties and short tail", np.tile(np.arange(-20, 20) * 0.0002, 4), True),
        ("BLAS with random unique queries", rng.normal(0, 0.004, 213), False),
    )
    for name, queries, tied in cases:
        references = rng.normal(0, 0.004, 9377).astype(np.float32)
        if tied:
            references[:5500] = np.resize([0, 0.004, -0.004, 0.008, -0.008], 5500)
        references = references[:, None]
        queries = np.asarray(queries, dtype=np.float32)[:, None]
        stock = faiss.IndexFlatL2(1)
        stock.add(references)
        _, expected = stock.search(queries, k)
        rewritten = _ReusedFlatL2(faiss, chunk_size=37, log=False)
        rewritten.add(references)
        _, observed = rewritten.search(queries, k)
        order = rng.permutation(len(queries))
        np.testing.assert_array_equal(observed[order], expected[order], err_msg=name)
        print(f"Matched stock FAISS labels: {name}", flush=True)


if __name__ == "__main__":
    verify()
