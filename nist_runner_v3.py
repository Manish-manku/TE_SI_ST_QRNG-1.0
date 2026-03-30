from __future__ import annotations

from dataclasses import dataclass
from math import floor, log2, sqrt
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
from scipy.special import erfc, gammaincc


TEST_NAMES = [
    "Frequency (Monobit)",
    "Block Frequency",
    "Runs",
    "Longest Run of Ones",
    "Binary Matrix Rank",
    "Discrete Fourier Transform",
    "Non-overlapping Template",
    "Overlapping Template",
    "Maurer's Universal",
    "Linear Complexity",
    "Serial",
    "Approximate Entropy",
    "Cumulative Sums",
    "Random Excursions",
    "Random Excursions Variant",
]
N_TESTS = len(TEST_NAMES)


@dataclass
class NISTResult:
    p_values: List[Optional[float]]
    passed: List[Optional[bool]]
    n_bits: int
    backend: str = "NIST SP800-22 (chunked v3)"

    def pass_rate(self) -> float:
        vals = [p for p in self.passed if p is not None]
        if not vals:
            return 0.0
        return float(np.mean(vals))


class NISTTestRunner:
    """Chunk-capable NIST SP 800-22 test runner.

    Streaming path accumulates cheap statistics for tests that naturally support
    online updates, while bounded reservoirs are used for tests requiring full
    sequence access.
    """

    def __init__(self, significance: float = 0.01, chunk_size: int = 1_000_000,
                 max_reservoir_bits: int = 4_000_000, seed: int = 7):
        self.significance = significance
        self.chunk_size = chunk_size
        self.max_reservoir_bits = max_reservoir_bits
        self._rng = np.random.default_rng(seed)

    def run_all(self, bits: Sequence[int] | np.ndarray) -> NISTResult:
        return self.run_all_chunked(self._array_chunks(bits, self.chunk_size), total_bits=len(bits))

    def run_all_chunked(self, chunks: Iterable[Sequence[int] | np.ndarray],
                        total_bits: Optional[int] = None) -> NISTResult:
        stream = _StreamState(max_reservoir_bits=self.max_reservoir_bits, rng=self._rng)
        for chunk in chunks:
            arr = np.asarray(chunk, dtype=np.uint8)
            if arr.size == 0:
                continue
            stream.update(arr)

        n = stream.n_bits if total_bits is None else total_bits
        if n < 1000 or stream.n_bits < 1000:
            return NISTResult([None] * N_TESTS, [None] * N_TESTS, n_bits=stream.n_bits)

        sample = stream.sample_bits()
        pvals = [
            self._frequency_monobit(stream),
            self._block_frequency(sample),
            self._runs(stream),
            self._longest_run_of_ones(sample),
            self._binary_matrix_rank(sample),
            self._discrete_fourier_transform(stream),
            self._non_overlapping_template_matching(sample),
            self._overlapping_template_matching(sample),
            self._maurers_universal(sample),
            self._linear_complexity(sample),
            self._serial(sample),
            self._approximate_entropy(sample),
            self._cumulative_sums(sample),
            self._random_excursions(sample),
            self._random_excursions_variant(sample),
        ]
        passed = [None if p is None else bool(p >= self.significance) for p in pvals]
        return NISTResult(p_values=pvals, passed=passed, n_bits=stream.n_bits)

    def _array_chunks(self, bits: Sequence[int] | np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
        arr = np.asarray(bits, dtype=np.uint8)
        for i in range(0, arr.size, chunk_size):
            yield arr[i:i + chunk_size]

    def _frequency_monobit(self, s: "_StreamState") -> Optional[float]:
        if s.n_bits == 0:
            return None
        sobs = abs(s.ones - s.zeros) / sqrt(s.n_bits)
        return float(erfc(sobs / sqrt(2.0)))

    def _block_frequency(self, bits: np.ndarray, m: int = 128) -> Optional[float]:
        n = bits.size
        if n < m:
            return None
        n_blocks = n // m
        b = bits[: n_blocks * m].reshape(n_blocks, m)
        pi = b.mean(axis=1)
        chi2 = 4.0 * m * np.sum((pi - 0.5) ** 2)
        return float(gammaincc(n_blocks / 2.0, chi2 / 2.0))

    def _runs(self, s: "_StreamState") -> Optional[float]:
        n = s.n_bits
        if n <= 1:
            return None
        pi = s.ones / n
        tau = 2.0 / sqrt(n)
        if abs(pi - 0.5) >= tau:
            return 0.0
        v_obs = s.runs
        num = abs(v_obs - 2.0 * n * pi * (1.0 - pi))
        den = 2.0 * sqrt(2.0 * n) * pi * (1.0 - pi)
        return float(erfc(num / den))

    def _longest_run_of_ones(self, bits: np.ndarray) -> Optional[float]:
        n = bits.size
        if n < 128:
            return None
        m, v, pi = 8, [1, 2, 3, 4], [0.2148, 0.3672, 0.2305, 0.1875]
        if n >= 6272:
            m, v, pi = 128, [4, 5, 6, 7, 8, 9], [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        if n >= 750000:
            m, v, pi = 10000, [10, 11, 12, 13, 14, 15, 16], [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]

        n_blocks = n // m
        if n_blocks == 0:
            return None
        b = bits[: n_blocks * m].reshape(n_blocks, m)
        max_runs = np.apply_along_axis(_max_ones_run, 1, b)

        bins = np.zeros(len(pi), dtype=np.int64)
        for r in max_runs:
            idx = np.searchsorted(v, r, side="right") - 1
            idx = np.clip(idx, 0, len(pi) - 1)
            bins[idx] += 1

        chi2 = np.sum((bins - n_blocks * np.asarray(pi)) ** 2 / (n_blocks * np.asarray(pi)))
        return float(gammaincc((len(pi) - 1) / 2.0, chi2 / 2.0))

    def _binary_matrix_rank(self, bits: np.ndarray) -> Optional[float]:
        m = q = 32
        block = m * q
        n_m = bits.size // block
        if n_m == 0:
            return None
        bits = bits[: n_m * block].reshape(n_m, m, q)
        full_rank = 0
        rank_m1 = 0
        for mat in bits:
            r = _binary_rank(mat)
            if r == m:
                full_rank += 1
            elif r == m - 1:
                rank_m1 += 1
        rem = n_m - full_rank - rank_m1
        p = [0.2888, 0.5776, 0.1336]
        f = np.array([full_rank, rank_m1, rem], dtype=np.float64)
        e = n_m * np.asarray(p)
        chi2 = np.sum((f - e) ** 2 / e)
        return float(np.exp(-chi2 / 2.0))

    def _discrete_fourier_transform(self, s: "_StreamState") -> Optional[float]:
        x = s.sample_bits()
        if x.size < 1000:
            return None
        y = 2 * x.astype(np.int8) - 1
        n = y.size
        m = np.abs(np.fft.fft(y))[: n // 2]
        t = np.sqrt(np.log(1.0 / 0.05) * n)
        n0 = 0.95 * n / 2.0
        n1 = np.sum(m < t)
        d = (n1 - n0) / np.sqrt(n * 0.95 * 0.05 / 4.0)
        return float(erfc(abs(d) / np.sqrt(2.0)))

    def _non_overlapping_template_matching(self, bits: np.ndarray) -> Optional[float]:
        m = 9
        if bits.size < 10_000:
            return None
        n_blocks = 8
        block_len = bits.size // n_blocks
        if block_len <= m:
            return None
        templates = [np.array([0,0,0,0,0,0,0,0,1],dtype=np.uint8), np.array([1,1,1,1,1,1,1,1,0],dtype=np.uint8)]
        pvals = []
        for tpl in templates:
            counts = []
            for i in range(n_blocks):
                b = bits[i * block_len:(i + 1) * block_len]
                cnt, j = 0, 0
                while j <= b.size - m:
                    if np.array_equal(b[j:j + m], tpl):
                        cnt += 1
                        j += m
                    else:
                        j += 1
                counts.append(cnt)
            counts = np.asarray(counts, dtype=np.float64)
            mu = (block_len - m + 1) / (2 ** m)
            var = block_len * ((1 / (2 ** m)) - ((2 * m - 1) / (2 ** (2 * m))))
            chi2 = np.sum((counts - mu) ** 2 / var)
            pvals.append(float(gammaincc(n_blocks / 2.0, chi2 / 2.0)))
        return float(np.mean(pvals))

    def _overlapping_template_matching(self, bits: np.ndarray) -> Optional[float]:
        m = 9
        if bits.size < 10_000:
            return None
        n_blocks = 8
        block_len = bits.size // n_blocks
        tpl = np.ones(m, dtype=np.uint8)
        counts = []
        for i in range(n_blocks):
            b = bits[i * block_len:(i + 1) * block_len]
            c = 0
            for j in range(0, b.size - m + 1):
                if np.array_equal(b[j:j + m], tpl):
                    c += 1
            counts.append(c)
        counts = np.asarray(counts)
        lam = (block_len - m + 1) / (2 ** m)
        probs = [np.exp(-lam)]
        for u in range(1, 5):
            probs.append(np.exp(-lam) * lam ** u / np.math.factorial(u))
        probs.append(max(0.0, 1.0 - np.sum(probs)))
        cats = np.zeros(6)
        for c in counts:
            cats[min(c, 5)] += 1
        expv = n_blocks * np.asarray(probs)
        chi2 = np.sum((cats - expv) ** 2 / np.maximum(expv, 1e-12))
        return float(gammaincc(5 / 2.0, chi2 / 2.0))

    def _maurers_universal(self, bits: np.ndarray) -> Optional[float]:
        L = 7
        Q = 1280
        K = bits.size // L - Q
        if K <= 0:
            return None
        tbl = np.zeros(2 ** L, dtype=np.int64)

        def to_int(segment):
            v = 0
            for bit in segment:
                v = (v << 1) | int(bit)
            return v

        for i in range(Q):
            tbl[to_int(bits[i * L:(i + 1) * L])] = i + 1

        s = 0.0
        for i in range(Q, Q + K):
            idx = to_int(bits[i * L:(i + 1) * L])
            d = i + 1 - tbl[idx]
            tbl[idx] = i + 1
            s += np.log2(d)

        fn = s / K
        expected = 6.1962507
        variance = 3.125
        sigma = np.sqrt(variance / K)
        return float(erfc(abs(fn - expected) / (np.sqrt(2.0) * sigma)))

    def _linear_complexity(self, bits: np.ndarray) -> Optional[float]:
        m = 500
        n_blocks = bits.size // m
        if n_blocks == 0:
            return None
        vals = []
        for i in range(n_blocks):
            block = bits[i * m:(i + 1) * m]
            l = _berlekamp_massey(block)
            mu = m / 2.0 + (9 + (-1) ** (m + 1)) / 36.0 - (m / 3.0 + 2 / 9.0) / (2 ** m)
            vals.append((-1) ** m * (l - mu) + 2 / 9.0)
        vals = np.asarray(vals)
        bins = np.array([
            np.sum(vals <= -2.5),
            np.sum((vals > -2.5) & (vals <= -1.5)),
            np.sum((vals > -1.5) & (vals <= -0.5)),
            np.sum((vals > -0.5) & (vals <= 0.5)),
            np.sum((vals > 0.5) & (vals <= 1.5)),
            np.sum((vals > 1.5) & (vals <= 2.5)),
            np.sum(vals > 2.5),
        ], dtype=np.float64)
        pi = np.array([0.0104, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.0208])
        chi2 = np.sum((bins - n_blocks * pi) ** 2 / (n_blocks * pi))
        return float(gammaincc(6 / 2.0, chi2 / 2.0))

    def _serial(self, bits: np.ndarray, m: int = 6) -> Optional[float]:
        if bits.size < m + 2:
            return None
        psim = _psi2(bits, m)
        psim1 = _psi2(bits, m - 1)
        psim2 = _psi2(bits, m - 2)
        d1 = psim - psim1
        d2 = psim - 2 * psim1 + psim2
        p1 = float(gammaincc((2 ** (m - 1)) / 2.0, d1 / 2.0))
        p2 = float(gammaincc((2 ** (m - 2)) / 2.0, d2 / 2.0))
        return float(min(p1, p2))

    def _approximate_entropy(self, bits: np.ndarray, m: int = 6) -> Optional[float]:
        n = bits.size
        if n < m + 1:
            return None

        def phi(mm):
            counts = np.zeros(2 ** mm, dtype=np.int64)
            mask = (1 << mm) - 1
            state = 0
            ext = np.concatenate([bits, bits[:mm - 1]])
            for i, b in enumerate(ext):
                state = ((state << 1) | int(b)) & mask
                if i >= mm - 1:
                    counts[state] += 1
            probs = counts / n
            nz = probs[probs > 0]
            return np.sum(nz * np.log(nz))

        apen = phi(m) - phi(m + 1)
        chi2 = 2.0 * n * (np.log(2) - apen)
        return float(gammaincc(2 ** (m - 1), chi2 / 2.0))

    def _cumulative_sums(self, bits: np.ndarray) -> Optional[float]:
        if bits.size < 2:
            return None
        x = 2 * bits.astype(np.int8) - 1
        s = np.cumsum(x)
        z = max(np.max(np.abs(s)), 1)
        n = bits.size
        # practical normal approximation of NIST formula
        p = erfc(z / np.sqrt(2 * n))
        return float(max(0.0, min(1.0, p)))

    def _random_excursions(self, bits: np.ndarray) -> Optional[float]:
        x = 2 * bits.astype(np.int8) - 1
        s = np.concatenate([[0], np.cumsum(x), [0]])
        cyc_idx = np.where(s == 0)[0]
        j = len(cyc_idx) - 1
        if j < 1:
            return None
        states = [-4, -3, -2, -1, 1, 2, 3, 4]
        pvals = []
        for st in states:
            visits = []
            for k in range(j):
                cycle = s[cyc_idx[k]:cyc_idx[k + 1] + 1]
                visits.append(np.sum(cycle == st))
            visits = np.asarray(visits)
            pi = np.array([0.5, 0.25, 0.125, 0.0625, 0.0312, 0.0313])
            v = np.array([np.sum(visits == u) for u in range(5)] + [np.sum(visits >= 5)])
            expv = j * pi
            chi2 = np.sum((v - expv) ** 2 / np.maximum(expv, 1e-12))
            pvals.append(float(gammaincc(5 / 2.0, chi2 / 2.0)))
        return float(np.min(pvals)) if pvals else None

    def _random_excursions_variant(self, bits: np.ndarray) -> Optional[float]:
        x = 2 * bits.astype(np.int8) - 1
        s = np.concatenate([[0], np.cumsum(x), [0]])
        j = np.sum(s == 0) - 1
        if j <= 0:
            return None
        pvals = []
        for st in list(range(-9, 0)) + list(range(1, 10)):
            ksi = np.sum(s == st)
            den = np.sqrt(2.0 * j * (4.0 * abs(st) - 2.0))
            pvals.append(float(erfc(abs(ksi - j) / den)))
        return float(np.min(pvals)) if pvals else None


class _StreamState:
    def __init__(self, max_reservoir_bits: int, rng: np.random.Generator):
        self.n_bits = 0
        self.ones = 0
        self.zeros = 0
        self.runs = 0
        self._prev: Optional[int] = None
        self._reservoir = np.empty(max_reservoir_bits, dtype=np.uint8)
        self._filled = 0
        self._max_res = max_reservoir_bits
        self._rng = rng

    def update(self, chunk: np.ndarray) -> None:
        chunk = chunk.astype(np.uint8, copy=False)
        self.n_bits += chunk.size
        self.ones += int(chunk.sum())
        self.zeros += int(chunk.size - chunk.sum())
        if chunk.size:
            if self._prev is None:
                self.runs = 1
            else:
                self.runs += int(chunk[0] != self._prev)
            self.runs += int(np.sum(chunk[1:] != chunk[:-1]))
            self._prev = int(chunk[-1])
        self._reservoir_sample(chunk)

    def _reservoir_sample(self, chunk: np.ndarray) -> None:
        for b in chunk:
            if self._filled < self._max_res:
                self._reservoir[self._filled] = b
                self._filled += 1
            else:
                j = self._rng.integers(0, self.n_bits)
                if j < self._max_res:
                    self._reservoir[j] = b

    def sample_bits(self) -> np.ndarray:
        return self._reservoir[:self._filled].copy()


def _max_ones_run(block: np.ndarray) -> int:
    maxr, cur = 0, 0
    for b in block:
        if b == 1:
            cur += 1
            if cur > maxr:
                maxr = cur
        else:
            cur = 0
    return maxr


def _binary_rank(mat: np.ndarray) -> int:
    a = mat.astype(np.uint8).copy()
    m, n = a.shape
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if a[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]
        for r in range(m):
            if r != row and a[r, col]:
                a[r, :] ^= a[row, :]
        row += 1
        rank += 1
        if row == m:
            break
    return rank


def _berlekamp_massey(bits: np.ndarray) -> int:
    n = bits.size
    c = np.zeros(n, dtype=np.uint8)
    b = np.zeros(n, dtype=np.uint8)
    c[0] = 1
    b[0] = 1
    l, m = 0, -1
    for n_idx in range(n):
        d = bits[n_idx]
        for i in range(1, l + 1):
            d ^= c[i] & bits[n_idx - i]
        if d == 1:
            t = c.copy()
            p = np.zeros(n, dtype=np.uint8)
            for j in range(n - (n_idx - m)):
                p[j + (n_idx - m)] = b[j]
            c ^= p
            if l <= n_idx / 2:
                l = n_idx + 1 - l
                m = n_idx
                b = t
    return int(l)


def _psi2(bits: np.ndarray, m: int) -> float:
    if m <= 0:
        return 0.0
    n = bits.size
    counts = np.zeros(2 ** m, dtype=np.int64)
    mask = (1 << m) - 1
    ext = np.concatenate([bits, bits[:m - 1]])
    state = 0
    for i, b in enumerate(ext):
        state = ((state << 1) | int(b)) & mask
        if i >= m - 1:
            counts[state] += 1
    return (2 ** m / n) * np.sum(counts ** 2) - n
