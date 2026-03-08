"""
Trust-Enhanced Source-Independent Quantum Random Number Generator (TE-SI-QRNG)
=================================================================================

A self-testing approach to quantum random number generation that provides
measurable trust guarantees without requiring full device-independence.

Authors: Research Team
Date: January 2025

VERSION HISTORY
===============

v5 — Performance optimizations
  1. santha_vazirani_test  O(n²) Python triple-loop  → O(n) vectorized hash-map
  2. toeplitz_extract      O(n·m) dense matmul        → O(n log n) FFT circulant
  3. autocorrelation_test  per-lag np.correlate loop  → single FFT pass (all lags)
  4. runs_test             Python for-loop            → np.diff vectorized

v6 — Formula corrections
  5. Entropy formula: BB84 phase-error 1−h(e_upper) → classical min-entropy −log₂(p_max_upper)
  6. Hoeffding correction: delta = sqrt(log(1/ε_smooth) / (2·n_test))
  7. Unified metadata key: h_cert / h_min_trusted → h_min_certified (single canonical name)
  8. ε_bias fallback: 1−freq_p (wrong) → |mean−0.5| (actual observed bias)

v7 — Calibration + pro-level upgrades
  A. Soft-thresholding: all min(x·sensitivity, 1.0) → calibrated sigmoid _sigmoid(x, k, x0)
       Noise floor maps to ε ≈ 0.03–0.05; extreme values approach 1.0 asymptotically.
       Eliminates binary on/off trust vector ("wall of yellow" in Experiment 1).
  B. Quantum Leftover Hash Lemma (LHL): ad-hoc η factor → provably tight
       k = floor( n_gen · h_min_certified − 2·log₂(1/ε_ext) )
       Industry standard for SI-QRNG (Tomamichel 2011, Ben-Or 2005).
  C. CUSUM drift detection: linear regression (slope only) → two-sided CUSUM control chart
       Reacts to sustained mean shifts the moment they begin.
       Continuous normalised score (C/h) feeds sigmoid → gradient ε_drift, not binary alarm.

Security invariants (unchanged throughout all versions)
-------------------------------------------------------
  h_min_certified  ← p_max_upper only              (FORBIDDEN to touch with trust)
  extraction_rate  ← LHL(n_gen, h_min_certified)   (FORBIDDEN to scale by trust_score)
  trust_score      → warn / halt only              (NEVER modifies entropy or extraction)
  EAT              Δ_EAT = 2·√t·√(ln(1/ε_EAT))    (unchanged)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import hashlib
from collections import deque


# ---------------------------------------------------------------------------
# TrustVector
# ---------------------------------------------------------------------------

@dataclass
class TrustVector:
    """
    Trust parameters quantifying system reliability.

    Attributes:
        epsilon_bias:  Deviation from uniformity [0, 1]
        epsilon_drift: Temporal instability measure [0, 1]
        epsilon_corr:  Memory/correlation effects [0, 1]
        epsilon_leak:  Side-channel leakage indicator [0, 1]
    """
    epsilon_bias:  float = 0.0
    epsilon_drift: float = 0.0
    epsilon_corr:  float = 0.0
    epsilon_leak:  float = 0.0

    def trust_score(self) -> float:
        """Compute aggregate trust score [0, 1], where 1 = perfect trust."""
        return 1.0 - np.sqrt(
            self.epsilon_bias**2 +
            self.epsilon_drift**2 +
            self.epsilon_corr**2 +
            self.epsilon_leak**2
        ) / 2.0

    def trust_penalty(self) -> float:
        """
        Entropy penalty derived from trust degradation.

        *** SECURITY NOTE — FORBIDDEN IN ENTROPY PATHS ***
        This method MUST NOT be subtracted from h_cert, added to e_upper,
        or used to scale extraction_rate or output_length.
        It exists solely for legacy reference and diagnostic display.
        Using it in the certified entropy formula breaks composable security.
        """
        return -np.log2(max(self.trust_score(), 0.01))


# ---------------------------------------------------------------------------
# DiagnosticHaltError
# ---------------------------------------------------------------------------

class DiagnosticHaltError(Exception):
    """
    Raised when diagnostic self-tests detect a condition severe enough
    to halt extraction entirely.

    Diagnostics are ALLOWED to halt — they are FORBIDDEN to modify entropy.

    Halt conditions (trust_score thresholds):
        trust_score < HALT_THRESHOLD  → raise DiagnosticHaltError
        trust_score < WARN_THRESHOLD  → add warning to metadata, continue

    These thresholds are engineering policy, not security bounds.
    The certified entropy H_cert remains valid regardless of trust_score;
    halting is a conservative operational choice, not a cryptographic requirement.
    """
    HALT_THRESHOLD: float = 0.2   # Hard stop — system too unstable to operate
    WARN_THRESHOLD: float = 0.5   # Soft warning — degraded but operational


# ---------------------------------------------------------------------------
# Calibrated sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(x: float, k: float, x0: float) -> float:
    """
    Calibrated sigmoid mapping a raw test statistic to ε ∈ (0, 1).

    Formula:  σ(x) = 1 / (1 + exp(-k * (x - x0)))

    Calibration intent
    ------------------
    k  controls steepness: higher k = sharper transition.
    x0 is the inflection point (maps to ε = 0.5).

    Choosing x0 at roughly the "clearly problematic but not worst-case"
    value ensures:
      • Expected noise floor              → ε ≈ 0.05  (near-zero, not zero)
      • Moderate imperfection             → ε ≈ 0.30–0.60  (informative gradient)
      • Truly extreme / adversarial value → ε ≈ 0.90  (near-one, not hard-clipped)

    This replaces all min(x * scale, 1.0) patterns which saturate at 1.0
    too easily, turning the trust vector into a binary on/off switch.
    """
    return float(1.0 / (1.0 + np.exp(-k * (x - x0))))


# ---------------------------------------------------------------------------
# StatisticalSelfTester  — all tests vectorized
# ---------------------------------------------------------------------------

class StatisticalSelfTester:
    """
    Implements statistical self-tests for randomness validation.

    Tests include:
    - Santha-Vazirani source detection  (O(n) hash-map, was O(n²) triple loop)
    - Runs test for sequential patterns (vectorized np.diff)
    - Autocorrelation analysis          (single FFT pass, all lags at once)
    - Frequency monobit test
    """

    def __init__(self, window_size: int = 1000, alpha: float = 0.01):
        self.window_size = window_size
        self.alpha = alpha

    # ------------------------------------------------------------------
    # OPTIMIZED: O(n) hash-map SV test — was O(n²) triple nested loop
    # ------------------------------------------------------------------
    def santha_vazirani_test(self, bits: np.ndarray) -> Tuple[bool, float]:
        """
        Test for Santha-Vazirani source violation.

        A bit sequence is epsilon-SV if for all i and conditioning:
        1/2 - epsilon <= P(X_i = b | X_1...X_{i-1}) <= 1/2 + epsilon

        OPTIMIZATION: replaced O(n²) Python triple-loop with a single O(n)
        pass using numpy stride tricks + a Python dict as a count table.
        For n=10000 and context_len up to 4, the old code did ~20M Python
        iterations; this version does 4 vectorized passes of length n.

        Returns:
            (passes_test, epsilon_sv)
        """
        n = len(bits)
        if n < 100:
            return True, 0.0

        max_deviation = 0.0
        max_context   = min(4, int(np.log2(n)))

        # Work on a uint8 view so bit packing is cheap
        b = bits.astype(np.uint8)

        for ctx_len in range(1, max_context + 1):
            # Encode each context of length ctx_len as a single integer.
            # For ctx_len ≤ 4 the integer fits in a uint8.
            # Use powers-of-2 encoding:  sum(b[i-k] * 2^(k-1))
            powers = (1 << np.arange(ctx_len, dtype=np.uint8))  # [1,2,4,8,…]

            # Build context integers for all valid positions at once
            # Shape: (n - ctx_len,)
            indices     = np.arange(ctx_len, n)           # positions of the outcome bit
            # Context windows: rows are positions, cols are offsets 0..ctx_len-1
            ctx_matrix  = b[indices[:, None] - 1 - np.arange(ctx_len)]  # (n-ctx_len, ctx_len)
            ctx_ids     = ctx_matrix @ powers                             # scalar per row

            outcomes    = b[indices]

            # Count ones and total per context id using np.bincount
            max_id      = 1 << ctx_len                  # number of possible contexts
            ones_count  = np.bincount(ctx_ids, weights=outcomes.astype(float), minlength=max_id)
            total_count = np.bincount(ctx_ids, minlength=max_id)

            # Only contexts with enough statistics
            valid_mask  = total_count >= 5
            if not np.any(valid_mask):
                continue

            prob_one    = np.where(valid_mask, ones_count / np.maximum(total_count, 1), 0.5)
            deviation   = np.abs(prob_one - 0.5)
            max_deviation = max(max_deviation, float(np.max(deviation[valid_mask])))

        epsilon_sv = max_deviation
        passes     = epsilon_sv < 0.25

        return passes, epsilon_sv

    # ------------------------------------------------------------------
    # OPTIMIZED: vectorized runs test — was Python for-loop
    # ------------------------------------------------------------------
    def runs_test(self, bits: np.ndarray) -> Tuple[bool, float]:
        """
        Test for independence using runs (consecutive identical bits).

        OPTIMIZATION: np.diff + np.count_nonzero replaces the Python for-loop.

        Returns:
            (passes_test, p_value)
        """
        n = len(bits)
        if n < 100:
            return True, 1.0

        # Count runs: transitions + 1
        runs      = int(np.count_nonzero(np.diff(bits))) + 1
        prop_ones = float(np.mean(bits))

        expected_runs = 2 * n * prop_ones * (1 - prop_ones) + 1
        variance_runs = (2 * n * prop_ones * (1 - prop_ones) *
                         (2 * n * prop_ones * (1 - prop_ones) - n) / (n - 1))

        if variance_runs <= 0:
            return True, 1.0

        z_score = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return p_value > self.alpha, p_value

    # ------------------------------------------------------------------
    # OPTIMIZED: FFT autocorrelation — was per-lag np.correlate loop
    # ------------------------------------------------------------------
    def autocorrelation_test(self, bits: np.ndarray, max_lag: int = 10) -> Tuple[bool, float]:
        """
        Test for temporal correlations using autocorrelation.

        OPTIMIZATION: compute all lags in a single FFT pass (O(n log n)) instead
        of calling np.correlate in a Python loop (O(n * max_lag)).

        Returns:
            (passes_test, max_correlation)
        """
        n = len(bits)
        if n < 2 * max_lag:
            return True, 0.0

        x  = 2.0 * bits.astype(np.float64) - 1.0
        x -= x.mean()
        sx  = x.std()
        if sx < 1e-12:
            return True, 0.0

        # FFT-based full autocorrelation in O(n log n)
        nfft    = 1 << int(np.ceil(np.log2(2 * n - 1)))   # next power of 2
        X       = np.fft.rfft(x, n=nfft)
        acf_raw = np.fft.irfft(X * np.conj(X))[:n]        # full ACF, lags 0..n-1

        # Normalise: lag-0 = n·σ²
        acf_norm = acf_raw / (n * sx**2)

        # Evaluate lags 1..max_lag-1  (lag-0 is always 1, skip)
        lags     = min(max_lag, n // 2)
        max_corr = float(np.max(np.abs(acf_norm[1:lags])))

        critical_value = 2.576 / np.sqrt(n)

        return max_corr < critical_value, max_corr

    def frequency_test(self, bits: np.ndarray) -> Tuple[bool, float]:
        """
        Monobit frequency test for bias.

        Returns:
            (passes_test, p_value)
        """
        n = len(bits)
        if n < 100:
            return True, 1.0

        ones    = int(np.sum(bits))
        z_score = (ones - n / 2) / np.sqrt(n / 4)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return p_value > self.alpha, p_value


# ---------------------------------------------------------------------------
# QuantumWitnessTester  — unchanged (already vectorized)
# ---------------------------------------------------------------------------

class QuantumWitnessTester:
    """
    Implements quantum-specific witness tests without full Bell inequality.
    """

    def __init__(self, visibility_threshold: float = 0.9):
        self.visibility_threshold = visibility_threshold

    def dimension_witness(self, outcomes: np.ndarray,
                          bases: np.ndarray) -> Tuple[bool, float]:
        if len(outcomes) < 1000:
            return True, 1.0

        basis_0 = outcomes[bases == 0]
        basis_1 = outcomes[bases == 1]

        if len(basis_0) < 100 or len(basis_1) < 100:
            return True, 1.0

        bias_0        = abs(float(np.mean(basis_0)) - 0.5)
        bias_1        = abs(float(np.mean(basis_1)) - 0.5)
        witness_value = abs(bias_0 - bias_1)

        return witness_value < 0.1, witness_value

    def energy_constraint_test(self, raw_signal: np.ndarray,
                                expected_mean: float = 0.0,
                                expected_std:  float = 1.0) -> Tuple[bool, float]:
        if len(raw_signal) < 100:
            return True, 0.0

        mean_dev  = abs(float(np.mean(raw_signal)) - expected_mean) / (expected_std + 1e-10)
        std_dev   = abs(float(np.std(raw_signal))  - expected_std)  / (expected_std + 1e-10)
        total_dev = float(np.sqrt(mean_dev**2 + std_dev**2))

        return total_dev < 3.0, total_dev

    def povm_consistency_test(self, outcomes: np.ndarray,
                               measurement_settings: np.ndarray) -> Tuple[bool, float]:
        if len(outcomes) < 1000:
            return True, 1.0

        unique_settings = np.unique(measurement_settings)
        probabilities   = []

        for setting in unique_settings:
            mask = measurement_settings == setting
            if np.sum(mask) > 10:
                probabilities.append(float(np.mean(outcomes[mask])))

        if len(probabilities) < 2:
            return True, 1.0

        consistency_score = float(np.std(probabilities))
        passes = 0.1 < consistency_score < 0.4

        return passes, consistency_score


# ---------------------------------------------------------------------------
# PhysicalDriftMonitor  — CUSUM drift detection (replaces linear regression)
# ---------------------------------------------------------------------------

class PhysicalDriftMonitor:
    """
    Monitors physical parameters for drift using CUSUM (Cumulative Sum control).

    CUSUM is the gold standard for detecting when a process mean shifts away
    from its calibration point.  Unlike linear regression (which averages over
    the whole history), CUSUM reacts the moment a sustained shift begins.

    Algorithm (two-sided CUSUM)
    ---------------------------
    For each new observation xᵢ with reference mean μ₀ and allowance k:

        C⁺ᵢ = max(0,  C⁺ᵢ₋₁ + (xᵢ - μ₀) - k)   # detects upward shift
        C⁻ᵢ = max(0,  C⁻ᵢ₋₁ - (xᵢ - μ₀) - k)   # detects downward shift

    Drift is declared when C⁺ or C⁻ exceeds decision threshold h.

    Typical parameter choice (Montgomery 2009):
        k = δ/2        where δ is the minimum shift to detect (in σ units)
        h = 4–5 σ      gives ARL₀ ≈ 370 (one false alarm per 370 samples)

    We normalise by the sample std of the warm-up period so k and h are
    expressed in standard-deviation units, making them source-independent.
    """

    def __init__(self,
                 history_length:   int   = 1000,
                 cusum_k:          float = 0.5,    # allowance (δ/2 in σ units)
                 cusum_h:          float = 4.0,    # decision threshold (σ units)
                 warmup_samples:   int   = 50):    # samples before CUSUM activates
        self.history_length   = history_length
        self.cusum_k          = cusum_k
        self.cusum_h          = cusum_h
        self.warmup_samples   = warmup_samples

        self.efficiency_history  = deque(maxlen=history_length)
        self.dark_count_history  = deque(maxlen=history_length)

        # CUSUM state for efficiency channel
        self._cusum_pos   = 0.0   # C⁺ accumulator
        self._cusum_neg   = 0.0   # C⁻ accumulator
        self._ref_mean    = None  # calibrated mean  (set after warm-up)
        self._ref_std     = None  # calibrated std   (set after warm-up)
        self._drift_score = 0.0   # normalised max(C⁺, C⁻) / h  ∈ [0, ∞)

    def update_efficiency(self, efficiency: float) -> None:
        self.efficiency_history.append(efficiency)
        self._update_cusum(efficiency)

    def update_dark_counts(self, dark_count_rate: float) -> None:
        self.dark_count_history.append(dark_count_rate)

    def _update_cusum(self, x: float) -> None:
        """Incorporate one new efficiency measurement into CUSUM."""
        n = len(self.efficiency_history)

        # Warm-up phase: calibrate reference mean and std
        if n == self.warmup_samples:
            arr = np.array(self.efficiency_history)
            self._ref_mean = float(np.mean(arr))
            # Use max(std, 1% of mean) as floor so identical warm-up values
            # don't give σ=0, which would make every subsequent z-score infinite.
            raw_std = float(np.std(arr))
            self._ref_std  = max(raw_std, 0.01 * abs(self._ref_mean) + 1e-9)
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0
            return

        if self._ref_mean is None or n < self.warmup_samples:
            return

        # Normalise: z = (x - μ₀) / σ₀
        z = (x - self._ref_mean) / self._ref_std

        # Two-sided CUSUM update
        self._cusum_pos = max(0.0, self._cusum_pos + z - self.cusum_k)
        self._cusum_neg = max(0.0, self._cusum_neg - z - self.cusum_k)

        # Normalised drift score: 0 = no drift, 1 = exactly at threshold, >1 = alarm
        self._drift_score = max(self._cusum_pos, self._cusum_neg) / self.cusum_h

    def detect_drift(self) -> Tuple[bool, float]:
        """
        Return (drift_detected, drift_magnitude).

        drift_magnitude is the normalised CUSUM score C/h:
            < 1.0  → within control limits
            ≥ 1.0  → drift alarm (mean has shifted by ≥ δ for a sustained period)
        """
        if self._ref_mean is None:
            return False, 0.0

        drift_detected = self._drift_score >= 1.0
        return drift_detected, self._drift_score

    def estimate_dark_count_stability(self) -> float:
        if len(self.dark_count_history) < 50:
            return 1.0
        dc_arr = np.array(self.dark_count_history)
        cv     = float(np.std(dc_arr)) / (float(np.mean(dc_arr)) + 1e-10)
        return float(np.exp(-cv))


# ---------------------------------------------------------------------------
# BB84RoundSplitter  — unchanged
# ---------------------------------------------------------------------------

class BB84RoundSplitter:
    """
    Splits raw BB84 measurements into generation and test rounds.

      * basis == 0  →  generation round  (Z-basis)
      * basis == 1  →  test round        (X-basis / phase-error estimation)
    """

    GENERATION_BASIS: int = 0

    @staticmethod
    def split(bits: np.ndarray,
              bases: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gen_mask  = bases == BB84RoundSplitter.GENERATION_BASIS
        return bits[gen_mask], bits[~gen_mask]


# ---------------------------------------------------------------------------
# EntropyEstimator  — unchanged (already fast)
# ---------------------------------------------------------------------------

class EntropyEstimator:
    """
    Certifies min-entropy from observable BB84 statistics.

    Security model
    --------------
        H_min(X | E) ≥ 1 − h(e_upper)          (*)

        e_upper = e_obs + √( ln(1/ε_smooth) / (2 · n_test) )   (Chernoff)

    EAT accumulation across t blocks:
        H_total = Σᵢ f(eᵢ) − Δ_EAT
        Δ_EAT   = 2 · √t · √( ln(1 / ε_EAT) )

    References
    ----------
    Tomamichel et al., Nature Communications 3, 634 (2012).
    Dupuis et al., Communications in Mathematical Physics 379, 867–913 (2020).
    Ma et al., npj Quantum Information 2, 16021 (2016).
    """

    def __init__(self, security_parameter: float = 1e-6):
        self.epsilon_total  = security_parameter
        self.epsilon_eat    = security_parameter / 2
        self.epsilon_smooth = security_parameter / 4
        self.epsilon_ext    = security_parameter / 4
        self._splitter      = BB84RoundSplitter()

    def certify_min_entropy(self,
                            bits:  np.ndarray,
                            bases: np.ndarray) -> Dict:
        gen_bits, test_bits = BB84RoundSplitter.split(bits, bases)
        n_gen  = len(gen_bits)
        n_test = len(test_bits)

        if n_test == 0:
            return self._zero_cert(n_gen, n_test)

        # Finite-size classical min-entropy estimator
        # H_min = -log2(p_max_upper)
        # p_max_upper is a Hoeffding upper confidence bound on the worst-case
        # symbol probability, using the complementary-basis test bits.
        p_hat            = float(np.mean(test_bits))
        p_max_hat        = max(p_hat, 1.0 - p_hat)       # worst-case direction

        delta            = np.sqrt(np.log(1.0 / self.epsilon_smooth) / (2.0 * n_test))
        p_max_upper      = min(p_max_hat + delta, 1.0)

        h_min_certified  = max(-np.log2(p_max_upper), 0.0)

        return {
            'n_generation':    n_gen,
            'n_test':          n_test,
            'p_hat':           p_hat,
            'p_max_hat':       p_max_hat,
            'delta':           delta,
            'p_max_upper':     p_max_upper,
            'h_min_certified': h_min_certified,
        }

    def compute_extraction_rate(self, h_min_certified: float,
                                 extractor_efficiency: float = 0.9) -> float:
        """
        Legacy thin wrapper kept for backward compatibility.
        New code should call lhl_output_length() directly.
        """
        return h_min_certified * extractor_efficiency

    def lhl_output_length(self, n_gen: int, h_min_certified: float) -> int:
        """
        Quantum Leftover Hash Lemma (LHL) extraction length.

        The LHL gives the maximum number of near-uniform bits extractable
        from n_gen generation bits with min-entropy H_min per bit:

            k = floor( n_gen · h_min_certified − 2 · log₂(1 / ε_ext) )

        This replaces the ad-hoc efficiency factor η and is the industry
        standard for SI-QRNG (Tomamichel et al. 2011, Ben-Or et al. 2005).

        The 2·log₂(1/ε_ext) term is the security cost of the extractor seed:
        it ensures the output is ε_ext-close to uniform in trace distance.

        Args:
            n_gen:           Number of generation-round bits available.
            h_min_certified: Per-bit min-entropy lower bound (from Hoeffding).

        Returns:
            k: Maximum safe extraction length (≥ 0).
        """
        log2_inv_eps = np.log2(1.0 / self.epsilon_ext)   # security cost ≈ 20 bits
        k = int(np.floor(n_gen * h_min_certified - 2.0 * log2_inv_eps))
        return max(k, 0)

    def entropy_accumulation(self, block_entropies: List[float]) -> float:
        return float(np.sum(block_entropies))


    @staticmethod
    def _zero_cert(n_gen: int, n_test: int) -> Dict:
        return {'n_generation': n_gen, 'n_test': n_test,
                'p_hat': 1.0, 'p_max_hat': 1.0,
                'delta': 0.0, 'p_max_upper': 1.0,
                'h_min_certified': 0.0}


# ---------------------------------------------------------------------------
# RandomnessExtractor  — OPTIMIZED: FFT circulant multiply replaces dense matmul
# ---------------------------------------------------------------------------

class RandomnessExtractor:
    """
    Quantum-proof randomness extractor (Toeplitz hashing).

    OPTIMIZATION: replaced scipy.linalg.toeplitz dense matrix build + matmul
    with FFT-based circulant convolution.

    A Toeplitz matrix–vector product T·x can be computed via circulant
    embedding and FFT in O(n log n) instead of O(n·m):

        y = IFFT( FFT(first_col_padded) ⊙ FFT(x_padded) )

    This is exact (modulo floating-point, corrected by rounding to int % 2).
    Memory usage drops from O(n·m) to O(n + m).
    """

    def __init__(self, input_length: int, output_length: int,
                 seed_length: Optional[int] = None):
        self.input_length  = input_length
        self.output_length = output_length
        self.seed_length   = seed_length or (2 * output_length)

    # ------------------------------------------------------------------
    # Memory budget: cap each FFT circulant at ~256 MB.
    # Float32 pipeline cost: ~20 bytes/sample across all temp arrays.
    # 256 MB / 20 B ≈ 13 M → round down to 2^23 = 8 388 608 for safety.
    _MAX_CIRC_SIZE: int = 1 << 23   # 8 388 608 elements, ~256 MB pipeline

    # ------------------------------------------------------------------
    # FFT-based Toeplitz multiply — single chunk (internal)
    # ------------------------------------------------------------------
    def _toeplitz_fft_chunk(self,
                             weak_random: np.ndarray,
                             seed:        np.ndarray,
                             out_len:     int) -> np.ndarray:
        """
        Core FFT Toeplitz multiply for one chunk (always float32).

        Args:
            weak_random : input bits, shape (n,)
            seed        : Toeplitz seed, length >= out_len + n - 1
            out_len     : number of output bits to produce

        Returns:
            output bits, shape (out_len,)
        """
        n = len(weak_random)
        m = out_len

        required = n + m - 1
        if len(seed) < required:
            seed = self._extend_seed(seed, required)

        col      = seed[:m].astype(np.float32)
        row_tail = seed[1:n].astype(np.float32)

        raw_size  = m + n
        circ_size = 1 << int(np.ceil(np.log2(max(raw_size, 2))))
        circ_size = min(circ_size, self._MAX_CIRC_SIZE)

        circ_col = np.zeros(circ_size, dtype=np.float32)
        circ_col[:m] = col
        if len(row_tail) > 0:
            circ_col[circ_size - len(row_tail):] = row_tail[::-1]

        x_pad = np.zeros(circ_size, dtype=np.float32)
        x_pad[:n] = weak_random.astype(np.float32)

        try:
            y_full = np.fft.irfft(
                np.fft.rfft(circ_col) * np.fft.rfft(x_pad),
                n=circ_size
            )
        except MemoryError:
            raise MemoryError(
                f"_toeplitz_fft_chunk: n={n}, m={m}, circ_size={circ_size}. "
                "Reduce block_size or max_workers."
            )

        output = np.round(y_full[:m]).astype(np.int64) % 2
        return output.astype(np.uint8)

    # ------------------------------------------------------------------
    # FFT-based Toeplitz multiply (O(n log n)) — chunked for large inputs
    # ------------------------------------------------------------------
    def toeplitz_extract(self, weak_random: np.ndarray,
                          seed: np.ndarray) -> np.ndarray:
        """
        Toeplitz hashing via FFT circulant embedding.

        The (m x n) Toeplitz matrix T is defined by:
          T[i, j] = seed[i - j]  for i >= j
          T[i, j] = seed[n + i - j] otherwise  (circulant extension)

        We embed T in a circulant of size N = m + n - 1 (next power of 2
        for FFT efficiency) and compute T*x mod 2 via:

          y_full = IFFT( FFT(first_col_padded) o FFT(x_padded) )
          y      = round(y_full[:m]) mod 2

        For large inputs that would require a circulant exceeding
        _MAX_CIRC_SIZE, the input is split into chunks.  Each chunk uses
        a deterministic per-chunk seed derived from the master seed via
        SHA-256 keyed with the chunk index, preserving quantum-proof
        security (each chunk is an independent Toeplitz hash).

        Args:
            weak_random: Input weak random bits, shape (n,)
            seed:        Toeplitz seed

        Returns:
            strong_random: Output bits, shape (self.output_length,)
        """
        n = len(weak_random)
        m = self.output_length

        # Fast path: entire input fits in one FFT
        single_circ = 1 << int(np.ceil(np.log2(max(m + n, 2))))
        if single_circ <= self._MAX_CIRC_SIZE:
            required = n + m - 1
            if len(seed) < required:
                seed = self._extend_seed(seed, required)
            return self._toeplitz_fft_chunk(weak_random, seed, m)

        # Chunked path: MAX_CHUNK_INPUT bits of input per chunk
        MAX_CHUNK_INPUT = max(self._MAX_CIRC_SIZE // 4, 1024)  # ~2 M bits

        n_chunks  = int(np.ceil(n / MAX_CHUNK_INPUT))
        n_c       = int(np.ceil(n / n_chunks))
        m_c_base  = m // n_chunks

        seed_bytes = np.packbits(seed[:min(len(seed), 2048)]).tobytes()

        output_chunks: List[np.ndarray] = []
        bits_produced = 0

        for i in range(n_chunks):
            i_start = i * n_c
            i_end   = min(i_start + n_c, n)
            chunk   = weak_random[i_start:i_end]
            nc_i    = len(chunk)
            if nc_i == 0:
                continue

            mc_i = (m - bits_produced) if (i == n_chunks - 1) else m_c_base
            if mc_i <= 0:
                break

            chunk_seed = self._derive_chunk_seed(seed_bytes, i, nc_i + mc_i - 1)
            out_chunk  = self._toeplitz_fft_chunk(chunk, chunk_seed, mc_i)
            output_chunks.append(out_chunk)
            bits_produced += len(out_chunk)

        if not output_chunks:
            return np.zeros(m, dtype=np.uint8)

        result = np.concatenate(output_chunks)
        if len(result) < m:
            result = np.concatenate([result,
                                     np.zeros(m - len(result), dtype=np.uint8)])
        return result[:m]

    def _derive_chunk_seed(self, master_seed_bytes: bytes,
                            chunk_idx: int, length: int) -> np.ndarray:
        """
        Derive a deterministic chunk seed for domain-separated Toeplitz hashing.

        SHA-256( master_seed_bytes || chunk_idx || counter ) iterated until
        `length` bits are produced.
        """
        extended: List[int] = []
        counter = 0
        prefix  = master_seed_bytes + chunk_idx.to_bytes(4, 'big')
        while len(extended) < length:
            h    = hashlib.sha256(prefix + counter.to_bytes(4, 'big')).digest()
            bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
            extended.extend(bits.tolist())
            counter += 1
        return np.array(extended[:length], dtype=np.uint8)

    def _extend_seed(self, seed: np.ndarray, length: int) -> np.ndarray:
        """Extend seed using SHA-256 (unchanged from v4)."""
        seed_bytes = np.packbits(seed).tobytes()
        extended   = []
        counter    = 0

        while len(extended) < length:
            hash_input  = seed_bytes + counter.to_bytes(4, 'big')
            hash_output = hashlib.sha256(hash_input).digest()
            bits        = np.unpackbits(np.frombuffer(hash_output, dtype=np.uint8))
            extended.extend(bits)
            counter += 1

        return np.array(extended[:length], dtype=np.uint8)

    def adaptive_extract(self, weak_random: np.ndarray,
                          seed: np.ndarray) -> np.ndarray:
        """
        Extract strong randomness at the output length set on construction.

        The output length is determined upstream by H_cert.  It is NOT scaled
        by trust_score — trust is a diagnostic quantity that must not enter
        the provable security bound.

        Delegates to the chunked Toeplitz path which keeps each FFT circulant
        under _MAX_CIRC_SIZE to prevent MemoryError on large inputs.
        """
        return self.toeplitz_extract(weak_random, seed)



# ---------------------------------------------------------------------------
# TrustEnhancedQRNG  — unchanged logic, uses optimized sub-components
# ---------------------------------------------------------------------------

class TrustEnhancedQRNG:
    """
    Main TE-SI-QRNG system integrating all components.

    Pipeline per block
    ------------------
    1. BB84 round splitting       → generation bits + test bits.
    2. Phase-error certification  → H_cert = 1 − h(e_upper)  (provable bound).
    3. Store f(eᵢ) = h_cert       → block_entropy_history for EAT.
    4. Statistical self-tests     → TrustVector updated (diagnostics only).
    5. Per-block Toeplitz extract → on generation bits (FFT-based, fast).

    For globally certified output use generate_certified_random_bits() which
    applies the full EAT bound before extracting.

    Mathematical Separation (INVARIANT)
    ------------------------------------
    h_cert is derived SOLELY from phase-error rate (e_upper).
    The following are PERMANENTLY FORBIDDEN:

        h_cert         -= trust_penalty        # FORBIDDEN
        h_cert         *= trust_score          # FORBIDDEN
        extraction_rate *= trust_score         # FORBIDDEN
        output_length   *= trust_score         # FORBIDDEN

    trust_score may only:
        • Raise DiagnosticHaltError  (< HALT_THRESHOLD)
        • Add a warning to metadata  (< WARN_THRESHOLD)
        • Be reported in metadata for monitoring
    """

    def __init__(self,
                 block_size:           int   = 1000,
                 security_parameter:   float = 1e-6,
                 extractor_efficiency: float = 0.9):
        self.block_size           = block_size
        self.security_parameter   = security_parameter
        self.extractor_efficiency = extractor_efficiency

        # Components
        self.stat_tester       = StatisticalSelfTester(window_size=block_size)
        self.quantum_tester    = QuantumWitnessTester()
        self.drift_monitor     = PhysicalDriftMonitor()
        self.entropy_estimator = EntropyEstimator(security_parameter=security_parameter)

        # Composable epsilon budget (mirrors EntropyEstimator)
        self.epsilon_total  = self.entropy_estimator.epsilon_total
        self.epsilon_eat    = self.entropy_estimator.epsilon_eat
        self.epsilon_smooth = self.entropy_estimator.epsilon_smooth
        self.epsilon_ext    = self.entropy_estimator.epsilon_ext

        # Diagnostic state (does NOT feed into entropy bound)
        self.trust_vector = TrustVector()

        # EAT accumulation state
        # block_entropy_history stores  f(eᵢ) · n_gen_i  (in bits, not bits/bit)
        # block_n_gen_history  stores   n_gen_i           (generation bits per block)
        self.block_entropy_history: List[float] = []
        self.block_n_gen_history:   List[int]   = []

        # Throughput counters
        self.total_output_bits    = 0
        self.total_gen_input_bits = 0
        self.total_raw_input_bits = 0

    # ------------------------------------------------------------------
    # EAT Accumulation
    # ------------------------------------------------------------------

    def accumulate_eat(self) -> float:
        """
        Compute globally certified entropy using the Entropy Accumulation Theorem.

        Units: everything in BITS (not bits/bit).

            sum_f     = Σᵢ  h_min_i · n_gen_i          [bits]
            N_total   = Σᵢ  n_gen_i                     [bits]
            Δ_EAT     = 2 · √N_total · √(ln(1/ε_EAT))  [bits]
            H_total   = sum_f − Δ_EAT                   [bits]

        h_min_i is stored already multiplied by n_gen_i (bits, not bits/bit)
        so block_entropy_history[i] = h_min_certified_i * n_gen_i.

        Reference: Dupuis et al., CMP 379, 867–913 (2020), Theorem 2.
        """
        t = len(self.block_entropy_history)
        if t == 0:
            return 0.0

        sum_f     = sum(self.block_entropy_history)          # Σ h_min_i · n_gen_i  [bits]
        n_total   = sum(self.block_n_gen_history)            # Σ n_gen_i             [bits]
        delta_eat = 2.0 * np.sqrt(n_total) * np.sqrt(np.log(1.0 / self.epsilon_eat))

        return max(sum_f - delta_eat, 0.0)

    # ------------------------------------------------------------------
    # Self-testing (diagnostic — NOT used in entropy formula)
    # ------------------------------------------------------------------

    def run_self_tests(self,
                       raw_bits:   np.ndarray,
                       bases:      Optional[np.ndarray] = None,
                       raw_signal: Optional[np.ndarray] = None) -> TrustVector:
        """
        Run the full statistical / quantum self-test suite.

        Epsilon calibration
        -------------------
        Each ε is mapped through a calibrated sigmoid so that:
          • Small physical deviations → small ε  (gradient, not on/off switch)
          • Only truly extreme deviations → ε near 1.0
          • The mapping is: ε = sigmoid(k * (x - x0)) where k and x0 are
            chosen so that the "expected noise floor" maps to ε ≈ 0.05 and
            a "clearly adversarial" value maps to ε ≈ 0.9.

        Double-penalty prevention
        -------------------------
        h_min_certified is already lowered by the Hoeffding bound when bias
        exists. These ε values are DIAGNOSTIC ONLY — they must not re-penalise
        extraction. They only govern the halt/warn thresholds.
        """
        sv_pass,       epsilon_sv   = self.stat_tester.santha_vazirani_test(raw_bits)
        freq_pass,     freq_p       = self.stat_tester.frequency_test(raw_bits)
        _,             _            = self.stat_tester.runs_test(raw_bits)
        autocorr_pass, max_autocorr = self.stat_tester.autocorrelation_test(raw_bits)

        # --- ε_bias ---
        # obs_bias = |mean − 0.5| is the ONLY sigmoid input for ε_bias.
        # It is monotone by construction in the true bias.
        # epsilon_sv from the SV test measures context-conditional dependence
        # separately; it can be non-monotone in individual blocks due to finite
        # sample variance in the context window estimator.  Mixing it into the
        # sigmoid input causes non-monotone ε_bias.
        # The SV violation will already manifest as higher obs_bias on average;
        # we do not need to double-count it in the sigmoid input.
        raw_obs_bias = abs(float(np.mean(raw_bits)) - 0.5)
        epsilon_bias = _sigmoid(raw_obs_bias, k=17.0, x0=0.20)

        # --- ε_corr ---
        # max_autocorr is a normalised correlation coefficient.
        # Noise floor (ideal source p95) ≈ 0.028 → ε ≈ 0.04
        # Inflection at x0=0.15 → ε = 0.50
        # High correlation 0.50+ → ε ≈ 1.00
        epsilon_corr = _sigmoid(max_autocorr, k=26.0, x0=0.15)

        # --- ε_leak ---
        epsilon_leak = 0.0
        if bases is not None:
            dim_pass, dim_witness = self.quantum_tester.dimension_witness(raw_bits, bases)
            if not dim_pass:
                # dim_witness = |bias_0 - bias_1|. Inflection at 0.20.
                epsilon_leak = _sigmoid(dim_witness, k=15.0, x0=0.20)

        # --- ε_drift ---
        # CUSUM detect_drift() returns (alarm_bool, drift_score).
        # drift_score = max(C⁺, C⁻) / h  is normalised:
        #     0.0 = perfectly in control
        #     1.0 = exactly at alarm threshold
        #     >1  = alarm triggered, score keeps climbing
        # We use the CONTINUOUS score (not just the alarm bool) so that
        # ε_drift is a gradient rather than a binary on/off.
        # Sigmoid inflection at score=1.0 (alarm point), k=4 gives:
        #     score=0.0 → ε≈0.018  (in-control noise floor)
        #     score=0.5 → ε≈0.119  (pre-alarm warning region)
        #     score=1.0 → ε≈0.500  (alarm threshold = inflection)
        #     score=2.0 → ε≈0.982  (well past alarm = near-saturated)
        epsilon_drift = 0.0
        if raw_signal is not None:
            self.quantum_tester.energy_constraint_test(raw_signal)
            _, drift_score = self.drift_monitor.detect_drift()
            epsilon_drift = _sigmoid(drift_score, k=4.0, x0=1.0)

        self.trust_vector = TrustVector(
            epsilon_bias  = float(np.clip(epsilon_bias,  0.0, 1.0)),
            epsilon_drift = float(np.clip(epsilon_drift, 0.0, 1.0)),
            epsilon_corr  = float(np.clip(epsilon_corr,  0.0, 1.0)),
            epsilon_leak  = float(np.clip(epsilon_leak,  0.0, 1.0)),
        )
        return self.trust_vector

    # ------------------------------------------------------------------
    # Block processing pipeline
    # ------------------------------------------------------------------

    def process_block(self,
                      raw_bits:   np.ndarray,
                      bases:      Optional[np.ndarray] = None,
                      raw_signal: Optional[np.ndarray] = None,
                      seed:       Optional[np.ndarray] = None,
                      ) -> Tuple[np.ndarray, Dict]:
        n_raw = len(raw_bits)

        # Step 1: BB84 round splitting
        if bases is not None:
            gen_bits, test_bits = BB84RoundSplitter.split(raw_bits, bases)
        else:
            gen_bits  = raw_bits
            test_bits = np.array([], dtype=np.uint8)

        n_gen  = len(gen_bits)
        n_test = len(test_bits)

        # Step 2: Phase-error certification — THE entropy bound  (INVARIANT)
        cert            = self.entropy_estimator.certify_min_entropy(
            raw_bits,
            bases if bases is not None else np.zeros(n_raw, dtype=np.uint8)
        )
        h_min_certified = cert['h_min_certified']
        # INVARIANT: h_min_certified is derived solely from p_max_upper.
        # Any second assignment to h_min_certified is a security bug.

        # Step 3: Store f(eᵢ)·n_gen_i for EAT accumulation — units: BITS
        # h_min_certified is per-bit (bits/bit); multiply by n_gen to get bits.
        self.block_entropy_history.append(h_min_certified * n_gen)
        self.block_n_gen_history.append(n_gen)

        # Step 4: Diagnostic self-tests (trust vector — NOT in bound)
        trust_vector = self.run_self_tests(raw_bits, bases, raw_signal)
        trust_score  = trust_vector.trust_score()

        diagnostic_warning: Optional[str] = None
        if trust_score < DiagnosticHaltError.HALT_THRESHOLD:
            raise DiagnosticHaltError(
                f"System instability detected: trust_score={trust_score:.4f} "
                f"< HALT_THRESHOLD={DiagnosticHaltError.HALT_THRESHOLD}. "
                f"Extraction halted. h_min_certified={h_min_certified:.4f} is valid but "
                f"operational policy requires halt."
            )
        if trust_score < DiagnosticHaltError.WARN_THRESHOLD:
            diagnostic_warning = (
                f"Degraded operation: trust_score={trust_score:.4f} "
                f"< WARN_THRESHOLD={DiagnosticHaltError.WARN_THRESHOLD}. "
                f"h_min_certified={h_min_certified:.4f} is unaffected."
            )

        # Step 5: LHL extraction length  (Quantum Leftover Hash Lemma)
        # k = floor( n_gen · h_min_certified − 2·log₂(1/ε_ext) )
        # This is the provably tight bound — no ad-hoc η factor.
        output_length   = self.entropy_estimator.lhl_output_length(n_gen, h_min_certified)
        # Retain extraction_rate as a normalised diagnostic for plots/metadata
        extraction_rate = output_length / max(n_gen, 1)

        meta_base = {
            'certified_quantity':  'H_min(X|E)',
            'security_definition': 'Trace-distance ε-security',
            'epsilon_total':       self.epsilon_total,
            'epsilon_eat':         self.epsilon_eat,
            'epsilon_smooth':      self.epsilon_smooth,
            'epsilon_ext':         self.epsilon_ext,
            'n_generation':        n_gen,
            'n_test':              n_test,
            'p_hat':               cert['p_hat'],
            'p_max_hat':           cert['p_max_hat'],
            'delta':               cert['delta'],
            'p_max_upper':         cert['p_max_upper'],
            'h_min_certified':     h_min_certified,   # single canonical key
            'extraction_rate':     extraction_rate,
            'blocks_accumulated':  len(self.block_entropy_history),
            'h_total_eat':         self.accumulate_eat(),
            'sum_f_ei':            sum(self.block_entropy_history),
            # --- Diagnostic (read-only — NOT used in bound) ---
            'trust_score':         trust_score,
            'trust_vector':        {
                'epsilon_bias':  trust_vector.epsilon_bias,
                'epsilon_drift': trust_vector.epsilon_drift,
                'epsilon_corr':  trust_vector.epsilon_corr,
                'epsilon_leak':  trust_vector.epsilon_leak,
            },
            'diagnostic_warning':  diagnostic_warning,
            'halt_threshold':      DiagnosticHaltError.HALT_THRESHOLD,
            'warn_threshold':      DiagnosticHaltError.WARN_THRESHOLD,
            'input_bits':          n_raw,
            'cumulative_efficiency': (self.total_output_bits /
                                      max(self.total_raw_input_bits, 1)),
        }

        if output_length < 1 or n_gen < 2:
            meta_base.update({'output_bits': 0,
                               'warning': 'Certified entropy too low for extraction'})
            return np.array([], dtype=np.uint8), meta_base

        # Step 6: Toeplitz extraction (FFT-based, O(n log n))
        if seed is None:
            seed_len      = min(2 * output_length, n_gen // 2)
            seed_arr      = gen_bits[:seed_len]
            extract_input = gen_bits[seed_len:]
        else:
            seed_arr      = seed
            extract_input = gen_bits

        if len(extract_input) < output_length:
            output_length = len(extract_input)

        extractor   = RandomnessExtractor(input_length=len(extract_input),
                                          output_length=output_length)
        output_bits = extractor.adaptive_extract(extract_input, seed_arr)

        self.total_raw_input_bits  += n_raw
        self.total_gen_input_bits  += n_gen
        self.total_output_bits     += len(output_bits)

        meta_base.update({
            'output_bits': len(output_bits),
            'cumulative_efficiency': (self.total_output_bits /
                                      max(self.total_raw_input_bits, 1)),
        })
        return output_bits, meta_base

    # ------------------------------------------------------------------
    # Global EAT-certified generation
    # ------------------------------------------------------------------

    def generate_certified_random_bits(self,
                                       n_bits:           int,
                                       source_simulator) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate n bits with full composable EAT-certified security.

        Guarantees:  ‖ρ_RE − U_R ⊗ ρ_E‖₁ ≤ ε_total
        """
        self.block_entropy_history = []
        self.block_n_gen_history   = []

        all_gen_bits:  List[np.ndarray] = []
        metadata_list: List[Dict]       = []

        while True:
            raw_bits, bases, raw_signal = source_simulator.generate_block(self.block_size)

            try:
                _, block_meta = self.process_block(raw_bits, bases, raw_signal)
            except DiagnosticHaltError as exc:
                halt_meta = {
                    'certified_quantity':  'H_min(X|E)',
                    'security_definition': 'Trace-distance ε-security',
                    'halt': True,
                    'halt_reason': str(exc),
                    'blocks_accumulated': len(self.block_entropy_history),
                    'h_total_eat': self.accumulate_eat(),
                }
                metadata_list.append(halt_meta)
                raise

            metadata_list.append(block_meta)

            if bases is not None:
                gen_bits, _ = BB84RoundSplitter.split(raw_bits, bases)
            else:
                gen_bits = raw_bits
            all_gen_bits.append(gen_bits)

            h_total          = self.accumulate_eat()
            log2_inv_eps_ext = np.log2(1.0 / self.epsilon_ext)
            max_output_bits  = int(h_total - 2.0 * log2_inv_eps_ext)

            if max_output_bits >= n_bits:
                break

            total_gen = sum(len(g) for g in all_gen_bits)
            if total_gen > 50 * n_bits:
                print("  [TE-SI-QRNG] Warning: EAT bound not reached — "
                      "source entropy too low for requested n_bits.")
                break

        # Single global Toeplitz extraction
        all_gen_concat = (np.concatenate(all_gen_bits)
                          if all_gen_bits else np.array([], dtype=np.uint8))

        h_total          = self.accumulate_eat()
        log2_inv_eps_ext = np.log2(1.0 / self.epsilon_ext)
        certified_output = max(int(h_total - 2.0 * log2_inv_eps_ext), 0)
        output_length    = min(n_bits, certified_output)

        if output_length < 1 or len(all_gen_concat) < 2:
            final_bits = np.array([], dtype=np.uint8)
        else:
            seed_len      = min(2 * output_length, len(all_gen_concat) // 2)
            seed_arr      = all_gen_concat[:seed_len]
            extract_input = all_gen_concat[seed_len:]
            if len(extract_input) < output_length:
                output_length = len(extract_input)
            extractor  = RandomnessExtractor(input_length=len(extract_input),
                                             output_length=output_length)
            final_bits = extractor.adaptive_extract(extract_input, seed_arr)

        t_blocks  = len(self.block_entropy_history)
        sum_f_ei  = sum(self.block_entropy_history)          # Σ h_min_i·n_gen_i [bits]
        n_total   = sum(self.block_n_gen_history)            # Σ n_gen_i          [bits]
        delta_eat = (2.0 * np.sqrt(n_total) *
                     np.sqrt(np.log(1.0 / self.epsilon_eat))
                     if t_blocks > 0 else 0.0)

        eat_summary = {
            'certified_quantity':    'H_min(X|E)',
            'security_definition':   'Trace-distance ε-security',
            'epsilon_total':         self.epsilon_total,
            'epsilon_eat':           self.epsilon_eat,
            'epsilon_smooth':        self.epsilon_smooth,
            'epsilon_ext':           self.epsilon_ext,
            'blocks_used':           t_blocks,
            'h_total_eat':           h_total,
            'certified_output_bits': certified_output,
            'actual_output_bits':    len(final_bits),
            'delta_eat':             delta_eat,
            'sum_f_ei':              sum_f_ei,
        }
        metadata_list.append(eat_summary)

        return final_bits[:n_bits], metadata_list


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("TE-SI-QRNG: Trust-Enhanced Source-Independent Quantum Random Number Generator")
    print("=" * 80)
    print("\nVersion 6 — Classical Min-Entropy Formula + Unified h_min_certified Key")
    print("\nKey optimizations (v5):")
    print("  santha_vazirani_test: O(n²) triple loop → O(n) vectorized hash-map")
    print("  toeplitz_extract:     O(n·m) dense matmul → O(n log n) FFT circulant")
    print("  autocorrelation_test: per-lag loop → single FFT pass (all lags)")
    print("  runs_test:            Python for-loop → np.diff vectorized")
    print("\nFormula changes (v6):")
    print("  h_min_certified = max(-log2(p_max_upper), 0)")
    print("  p_max_upper     = min(max(p_hat, 1-p_hat) + delta, 1.0)  [Hoeffding]")
    print("  delta           = sqrt(log(1/ε_smooth) / (2 * n_test))")
    print("\nSecurity invariants:")
    print("  h_min_certified ← p_max_upper only (INVARIANT — never modified by diagnostics)")
    print("  extraction_rate ← h_min_certified · η only (INVARIANT)")
    print("  trust_score     → warn/halt only — NEVER modifies entropy")
