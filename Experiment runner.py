"""
Experiment — Computational Overhead, Latency, Memory, Throughput, and
Scalability Evaluation of the TE-SI-QRNG Pipeline
=============================================================================

Measurement-only. Does NOT modify D_v16.py, New_simulator_v9.py, or any
other project file, and does NOT alter the TE-SI-QRNG security model.
Referred to everywhere simply as "Experiment" (never "Experiment 7/8").


===========================================================================
ARCHITECTURE MAP  (spec section 31 — built before any benchmarking code)
===========================================================================

  Input (params)
    -> QuantumSourceSimulator.generate_block()          [New_simulator_v9.py]
    -> GeneratedBlock(bits, bases, raw_signal)
        |
        v
  TrustEnhancedQRNG.process_block()                     [D_v16.py orchestrator]
        |
        +-- _certify_block()      [ BB84 split + Hoeffding/min-entropy
        |                           (EntropyEstimator.certify_min_entropy)
        |                           + EAT history append (session.append_block)]
        |
        +-- _run_diagnostics()    [PURE trust-monitoring: calls
        |                           run_self_tests() -> TrustVector,
        |                           then halt/warn decision against
        |                           DiagnosticHaltError thresholds]
        |
        +-- _extract_block()      [PURE extraction: EntropyEstimator.
        |                           lhl_output_length() + RandomnessExtractor.
        |                           toeplitz_extract() (FFT-circulant)]
        |
        +-- _assemble_metadata()  [bookkeeping: BlockMetadata dict +
                                    throughput counters]

  Multi-block loop: CertifiedGenerationSession.run() (via the
  generate_certified_random_bits() shim), driving process_block() per
  block and calling QRNGSessionState.accumulate_eat() (EAT) after every
  block, until the EAT-derived certified_output_bits >= n_bits requested.

IMPORTANT: _certify_block() is the certification-plane step (BB84 split,
Hoeffding min-entropy certification, EAT history append).The only purely
"trust monitoring" step in the live block pipeline is _run_diagnostics().
This benchmark times _certify_block() and _run_diagnostics() separately so
the two planes stay distinguishable in the data.

Also: run_self_tests() calls autocorrelation_test(), santha_vazirani_test()
and runs_test(); epsilon_corr is the max-fusion of the three signals.
epsilon_bias comes from an inline np.mean() calculation, epsilon_leak from
QuantumWitnessTester.dimension_witness(), and epsilon_drift from
energy_constraint_test() + PhysicalDriftMonitor (CUSUM). frequency_test() is
defined in StatisticalSelfTester but not called in the live path. Live tests
are labeled "trust_monitoring_live"; frequency_test is labeled
"trust_monitoring_defined_not_called".

Dense Toeplitz: RandomnessExtractor has exactly one extraction code path
(toeplitz_extract -> _toeplitz_fft_chunk, FFT-circulant, auto-chunked above
_MAX_CIRC_SIZE = 2**23). No dense O(n*m) matrix-multiply implementation
exists anywhere in D_v16.py. Recorded as status="not_present_in_codebase",
never benchmarked, never extrapolated (spec sections 3, 10).


===========================================================================
MEASUREMENT METHODOLOGY NOTES  (spec sections 5-9, and the earlier review's
valid corrections)
===========================================================================

- Timer: time.perf_counter() (monotonic, high-resolution) for wall-clock;
  time.process_time() (CPU-seconds actually consumed by this process) as
  the PRIMARY CPU metric. psutil per-process cpu_percent() is also recorded
  as a SECONDARY metric, with the documented caveat that it is noisy for
  sub-millisecond calls and can exceed 100% due to multi-threaded BLAS/FFT.

- Memory: three distinct, separately-labeled metrics, per the earlier
  review's correction —
    peak_python_tracked_MB : tracemalloc peak (Python-level allocator;
                              NumPy array allocations ARE visible to
                              tracemalloc since NumPy 1.15, but native
                              scratch memory inside C-level FFT/BLAS
                              routines is NOT).
    peak_rss_sampled_MB    : a background thread polls the process's RSS
                              every ~2 ms for the duration of the call and
                              reports the observed peak minus the
                              pre-call baseline. This is closer to "peak
                              memory during the operation" than a simple
                              before/after delta, but is still a sampled
                              approximation (a spike shorter than the
                              poll interval can be missed) — disabled for
                              known-O(1) components to avoid thread
                              overhead contaminating microsecond-scale
                              timings.
    rss_delta_MB            : simple before/after RSS delta (can be ~0 even
                              for real work, since temporaries are often
                              freed before the "after" sample).
  None of these three is asserted to be "total peak process memory."

- Cold vs. steady-state (the earlier review's primary correction): every
  end-to-end process_block() measurement is taken in BOTH modes and
  labeled accordingly —
    end_to_end_cold   : a FRESH TrustEnhancedQRNG instance (and fresh
                         session state) per repetition -> measures
                         independent single-block computational cost.
    end_to_end_steady : ONE TrustEnhancedQRNG instance reused across all
                         repetitions -> measures operational steady-state
                         cost (PhysicalDriftMonitor's CUSUM state
                         genuinely persists
                         across calls on the same instance; this is by
                         design in the real system, so steady-state is a
                         legitimate second measurement, not a bug).
  The internal 4-method breakdown is taken in COLD mode only (each of the
  four sub-benchmarks gets its own fresh TrustEnhancedQRNG + fresh block
  per repetition; any state the target method needs from earlier steps is
  computed UNTIMED immediately beforehand, so only the target method's own
  cost is inside the timed region).

- generate_certified_random_bits() (the full multi-block pipeline) is now
  REPEATED (not single-shot, per the earlier review's correction), with a
  fresh QuantumSourceSimulator + fresh TrustEnhancedQRNG per repetition,
  and full mean/median/std/min/max statistics reported.

- Every repeated measurement is preceded by untimed warm-up call(s); repeat
  counts shrink as size grows so a large size cannot silently balloon total
  run time; every call is wrapped in a SIGALRM wall-clock timeout (POSIX
  only) and a MemoryError handler, and unsafe sizes are recorded with
  status="skipped" and an exact reason rather than dropped silently.

- Percentile statistics (p95/p99) are only reported when there are enough
  repetitions to make them meaningful (n>=5 for p95, n>=10 for p99);
  otherwise the field is null, not a fabricated number from too few samples.

- All plotting, JSON/CSV/Markdown writing, and console printing happen
  strictly after all timing loops complete; wall-clock time for the
  "measurement phase" vs. the "reporting phase" is recorded separately in
  system_info so this separation is verifiable in the output, not just
  asserted in a comment.


===========================================================================
WHAT THIS DOES NOT ESTABLISH  (spec sections 20-22, 24)
===========================================================================

This is a software/simulation performance evaluation. It does not measure,
and this file's outputs must not be described as measuring: physical
SI-QRNG hardware, optical interference, detector noise, SPAD dead time,
physical side-channel injection, laboratory environmental drift, quantum
security, or composable security. NIST-style statistical validation is a
separate, already-existing experiment (experiment_6_nist_validation_v3.py)
and is not touched or re-run here. Any real-time deployment discussion in
the generated reports is explicitly labeled "deployment analysis /
extrapolation" against a stated target rate, never a hardware claim.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
import platform
import signal
import statistics
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import scipy
import psutil
import matplotlib
matplotlib.use("Agg")
import matplotlib
import matplotlib.pyplot as plt

from D_v16 import (
    TrustEnhancedQRNG,
    StatisticalSelfTester,
    QuantumWitnessTester,
    PhysicalDriftMonitor,
    TrustVector,
    EntropyEstimator,
    RandomnessExtractor,
    QRNGSessionState,
    DiagnosticHaltError,
    InsufficientEntropyError,
)
from New_simulator_v9 import QuantumSourceSimulator, IdealParams

_PROCESS = psutil.Process(os.getpid())
TIMING_ONLY = os.environ.get("TIMING_ONLY", "0") == "1"   # 1 = no tracemalloc, no RSS sampler
1


# ===========================================================================
# Configuration (spec section 13 — reproducibility)
# ===========================================================================

@dataclass
class ExperimentConfig:
    output_dir: str = "experiment"
    quick: bool = False
    safety_fraction: float = 0.5
    timeout_seconds: Optional[float] = 180.0
    seed: int = 42
    self_test_sizes: List[int] = field(default_factory=lambda: [10_000, 100_000, 1_000_000])
    cert_sizes: List[int] = field(default_factory=lambda: [10_000, 100_000, 1_000_000])
    eat_block_counts: List[int] = field(default_factory=lambda: [1, 10, 50, 100, 500])
    toeplitz_size_pairs: List[Tuple[int, int]] = field(default_factory=lambda: [
        (10_000, 5_000), (100_000, 50_000), (1_000_000, 500_000),
        (1_620_000, 1_600_000), (5_000_000, 2_000_000), (10_000_000, 4_000_000)])
    candidate_block_sizes: List[int] = field(default_factory=lambda: [
        10_000, 100_000, 1_000_000, 3_240_000, 3_500_000, 10_000_000])
    full_gen_targets: List[int] = field(default_factory=lambda: [100_000, 1_000_000])
    dense_toeplitz_benchmark_enabled: bool = False  # always False: no dense
    # implementation exists in the codebase (see architecture map above);
    # this flag is kept, set False, and documented per spec section 13's
    # explicit request for a "dense benchmark enabled/disabled" config
    # field, rather than silently omitting it.

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ===========================================================================
# Safety infrastructure
# ===========================================================================

class TimeoutSkipped(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: Optional[float]):
    if seconds is None or os.name != "posix":
        yield
        return

    def _handler(signum, frame):
        raise TimeoutSkipped(f"exceeded {seconds:.0f}s wall-clock timeout")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(max(seconds, 1)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def estimate_pipeline_memory_bytes(block_size: int) -> int:
    """Conservative, deliberately over-generous upper bound. Used only to
    decide skip/no-skip; never reported as a measured value (spec section 17
    concern from the earlier review)."""
    raw_bytes = block_size * (1 + 1 + 8)
    max_circ_elems = 1 << 23
    circ_elems = min(max_circ_elems, max(block_size * 2, 1024))
    fft_bytes = circ_elems * 16 * 4
    return int((raw_bytes + fft_bytes) * 3)


def check_memory_safety(bytes_needed: int, safety_fraction: float) -> Tuple[bool, str]:
    available = psutil.virtual_memory().available
    budget = available * safety_fraction
    if bytes_needed <= budget:
        return True, ""
    return False, (
        f"insufficient_memory: estimated need ~{bytes_needed/1e6:.0f} MB, "
        f"only {budget/1e6:.0f} MB available under safety_fraction="
        f"{safety_fraction} (total available: {available/1e6:.0f} MB)")


def repetitions_for_size(n: int) -> int:
    if n <= 100_000:
        return 5
    if n <= 1_000_000:
        return 3
    if n <= 5_000_000:
        return 2
    return 1


def warmup_for_size(n: int) -> int:
    return 1


# ===========================================================================
# Peak-RSS-during-call sampler
# ===========================================================================

class RSSPeakSampler:
    """Background-thread RSS poller. Only used for calls expected to run
    long enough (>~ a few ms) that thread-scheduling overhead doesn't
    meaningfully contaminate the timing; disabled by the caller for known
    O(1)/microsecond components."""

    def __init__(self, proc: psutil.Process, interval_s: float = 0.002):
        self.proc = proc
        self.interval = interval_s
        self._peak = None
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._peak = self.proc.memory_info().rss
        self._stop_flag = False

        def _poll():
            while not self._stop_flag:
                try:
                    rss = self.proc.memory_info().rss
                    if rss > self._peak:
                        self._peak = rss
                except Exception:
                    pass
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[int]:
        self._stop_flag = True
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak


# ===========================================================================
# Core repeated-measurement primitive
# ===========================================================================

def run_repeated(fn: Callable, n_repeats: int, warmup: int = 1,
                  timeout_seconds: Optional[float] = None,
                  sample_rss: bool = False, rss_interval_s: float = 0.002,
                  track_mem: bool = True) -> Dict:
    """
    Times fn() over n_repeats calls (after `warmup` untimed calls).
    Returns status/error/successful_runs/failed_runs plus raw per-call
    sample lists for: latency_ms, cpu_time_ms (process_time, PRIMARY cpu
    metric), cpu_percent (SECONDARY), peak_python_MB (tracemalloc),
    peak_rss_sampled_MB (background-thread sampled, or None if
    sample_rss=False), rss_delta_MB (simple before/after).
    """
    if TIMING_ONLY:
        sample_rss = False
        track_mem = False
    try:
        with time_limit(timeout_seconds):
            for _ in range(warmup):
                fn()
    except (TimeoutSkipped, MemoryError) as exc:
        return {"status": "skipped", "error": f"failed during warmup: {exc}",
                "successful_runs": 0, "failed_runs": 0, "samples": None}

    lat_ms, cpu_t_ms, cpu_p, peak_py, peak_rss, rss_d = [], [], [], [], [], []
    failed = 0

    for _ in range(n_repeats):
        gc.collect()
        if track_mem:
            tracemalloc.start()
        rss_before = _PROCESS.memory_info().rss
        _PROCESS.cpu_percent(interval=None)
        sampler = RSSPeakSampler(_PROCESS, rss_interval_s) if sample_rss else None
        if sampler:
            sampler.start()

        t0_cpu = time.process_time()
        try:
            with time_limit(timeout_seconds):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
        except (TimeoutSkipped, MemoryError) as exc:
            if sampler:
                sampler.stop()
            if track_mem:
                tracemalloc.stop()
            failed += 1
            continue
        t1_cpu = time.process_time()

        cpu_pct = _PROCESS.cpu_percent(interval=None)
        rss_after = _PROCESS.memory_info().rss
        peak_rss_val = sampler.stop() if sampler else None
        if track_mem:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_py.append(peak / (1024 ** 2))

        lat_ms.append((t1 - t0) * 1000.0)
        cpu_t_ms.append((t1_cpu - t0_cpu) * 1000.0)
        cpu_p.append(cpu_pct)
        if peak_rss_val is not None:
            peak_rss.append(max(peak_rss_val - rss_before, 0) / (1024 ** 2))
        rss_d.append((rss_after - rss_before) / (1024 ** 2))

    successful = n_repeats - failed
    if successful == 0:
        return {"status": "skipped", "error": "all repetitions failed (timeout/MemoryError)",
                "successful_runs": 0, "failed_runs": failed, "samples": None}

    return {
        "status": "ok" if failed == 0 else "partial",
        "error": None if failed == 0 else f"{failed}/{n_repeats} repetitions failed",
        "successful_runs": successful, "failed_runs": failed,
        "samples": {
            "latency_ms": lat_ms, "cpu_time_ms": cpu_t_ms, "cpu_percent": cpu_p,
            "peak_python_MB": peak_py,
            "peak_rss_sampled_MB": peak_rss if peak_rss else None,
            "rss_delta_MB": rss_d,
        },
    }


def stat_block(samples: Optional[List[float]]) -> Optional[Dict]:
    if not samples:
        return None
    n = len(samples)
    mean = statistics.mean(samples)
    out = {
        "mean": mean, "median": statistics.median(samples),
        "std": statistics.pstdev(samples) if n > 1 else 0.0,
        "min": min(samples), "max": max(samples), "n": n,
        "cv": (statistics.pstdev(samples) / mean) if (n > 1 and mean) else (0.0 if n > 1 else None),
        "p95": float(np.percentile(samples, 95)) if n >= 5 else None,
        "p99": float(np.percentile(samples, 99)) if n >= 10 else None,
    }
    return out


def make_record(component: str, category: str, run_result: Dict,
                 block_size: Optional[int] = None, extra_param: Optional[Dict] = None,
                 repetitions_requested: Optional[int] = None, warmup: Optional[int] = None,
                 throughput_Mbits_per_s: Optional[float] = None, notes: str = "") -> Dict:
    rec = {
        "component": component, "category": category,
        "block_size": block_size, "extra_param": extra_param or {},
        "repetitions_requested": repetitions_requested, "warmup": warmup,
        "successful_runs": run_result.get("successful_runs", 0),
        "failed_runs": run_result.get("failed_runs", 0),
        "status": run_result["status"], "error": run_result.get("error"),
        "throughput_Mbits_per_s": throughput_Mbits_per_s, "notes": notes,
    }
    samples = run_result.get("samples")
    if samples:
        rec["latency_ms"] = stat_block(samples["latency_ms"])
        rec["cpu_time_ms"] = stat_block(samples["cpu_time_ms"])
        rec["cpu_percent"] = stat_block(samples["cpu_percent"])
        rec["peak_python_tracked_MB"] = stat_block(samples["peak_python_MB"])
        rec["peak_rss_sampled_MB"] = stat_block(samples["peak_rss_sampled_MB"])
        rec["rss_delta_MB"] = stat_block(samples["rss_delta_MB"])
    else:
        for k in ("latency_ms", "cpu_time_ms", "cpu_percent",
                   "peak_python_tracked_MB", "peak_rss_sampled_MB", "rss_delta_MB"):
            rec[k] = None
    return rec


def run_indexed_benchmark(component: str, category: str, block_size: Optional[int],
                           n_total_needed: int, build_fixture_fn: Callable[[int], Any],
                           call_fn: Callable[[Any], None], repeats: int, warmup: int,
                           cfg: ExperimentConfig, benchmarks: List[Dict],
                           extra_param: Optional[Dict] = None, notes: str = "",
                           sample_rss: bool = False) -> Dict:
    """
    Builds n_total_needed fresh fixtures (untimed, wrapped in the same
    memory/timeout safety net), then times only call_fn(fixture) for each,
    via run_repeated(). Used for every 'cold'/fresh-state measurement in
    this file so independent-trial semantics are structurally guaranteed
    rather than merely intended.
    """
    try:
        with time_limit(cfg.timeout_seconds):
            fixtures = [build_fixture_fn(i) for i in range(n_total_needed)]
    except (TimeoutSkipped, MemoryError) as exc:
        rec = make_record(component, category,
                           {"status": "skipped", "error": f"fixture build failed: {exc}",
                            "successful_runs": 0, "failed_runs": 0, "samples": None},
                           block_size, extra_param, repeats, warmup, notes=notes)
        benchmarks.append(rec)
        return rec

    idx = {"i": 0}

    def _call():
        item = fixtures[idx["i"]]
        idx["i"] += 1
        call_fn(item)

    run_result = run_repeated(_call, n_repeats=repeats, warmup=warmup,
                               timeout_seconds=cfg.timeout_seconds, sample_rss=sample_rss)
    rec = make_record(component, category, run_result, block_size, extra_param,
                       repeats, warmup, notes=notes)
    benchmarks.append(rec)
    return rec


# ===========================================================================
# System / environment info (spec sections 5-6)
# ===========================================================================

def get_cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.strip().lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        p = platform.processor()
        return p if p else "unknown (platform.processor() returned empty string)"
    except Exception as exc:
        return f"unknown (could not read CPU model: {exc})"


def get_system_info() -> Dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_model": get_cpu_model(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_GB": round(vm.total / (1024 ** 3), 2),
        "available_ram_GB_at_start": round(vm.available / (1024 ** 3), 2),
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "psutil_version": psutil.__version__,
        "matplotlib_version": matplotlib.__version__,
        "posix_timeout_guard_active": os.name == "posix",
        "process_pid": os.getpid(),
    }


# ===========================================================================
# Functional correctness self-check (spec section 25, item 17)
# ===========================================================================

def functional_correctness_check(benchmarks: List[Dict]) -> Dict:
    """One untimed, non-benchmark sanity check: does toeplitz_extract()
    still return output of the requested length, dtype uint8, values in
    {0,1}? This guards against the instrumentation itself having silently
    broken the algorithm under test. Not a timing measurement."""
    rng = np.random.RandomState(123)
    n_gen, out_len = 20_000, 8_000
    weak = rng.randint(0, 2, size=n_gen).astype(np.uint8)
    seed = rng.randint(0, 2, size=min(2 * out_len, 4096)).astype(np.uint8)
    extractor = RandomnessExtractor(input_length=n_gen, output_length=out_len)
    try:
        out = extractor.toeplitz_extract(weak, seed)
        ok = (len(out) == out_len and out.dtype == np.uint8
              and bool(np.all((out == 0) | (out == 1))))
        result = {"check": "toeplitz_extract_functional_correctness",
                  "passed": bool(ok), "output_length_expected": out_len,
                  "output_length_actual": int(len(out)),
                  "output_dtype": str(out.dtype), "error": None}
    except Exception as exc:
        result = {"check": "toeplitz_extract_functional_correctness",
                  "passed": False, "error": str(exc)}
    print(f"\n[Functional correctness check] "
          f"{'PASSED' if result['passed'] else 'FAILED'}: {result}")
    return result


# ===========================================================================
# Section A — Trust-monitoring components
# ===========================================================================

def benchmark_trust_monitoring(cfg: ExperimentConfig, benchmarks: List[Dict]) -> None:
    print("\n[A] Trust-monitoring components")
    print("    trust_monitoring_live      = actually invoked by run_self_tests()")
    print("    trust_monitoring_defined_not_called = defined in StatisticalSelfTester "
          "but not called by run_self_tests() (see architecture map)")

    stat_tester = StatisticalSelfTester()
    quantum_tester = QuantumWitnessTester()
    rng = np.random.RandomState(cfg.seed)

    for n in cfg.self_test_sizes:
        r, w = repetitions_for_size(n), warmup_for_size(n)
        bits = rng.randint(0, 2, size=n).astype(np.uint8)
        bases = rng.randint(0, 2, size=n).astype(np.uint8)
        raw_signal = rng.randn(n)
        print(f"  n={n:,}  (repeats={r}, warmup={w}) ...")

        res = run_repeated(lambda: abs(float(np.mean(bits)) - 0.5), r, w,
                            cfg.timeout_seconds, sample_rss=False)
        benchmarks.append(make_record("epsilon_bias_raw_mean_calc", "trust_monitoring_live",
                                       res, n, repetitions_requested=r, warmup=w,
                                       notes="Inline np.mean()-based bias calc in "
                                             "run_self_tests(); not a StatisticalSelfTester method."))

        res = run_repeated(lambda: stat_tester.autocorrelation_test(bits), r, w,
                            cfg.timeout_seconds, sample_rss=(n >= 100_000))
        benchmarks.append(make_record("autocorrelation_test", "trust_monitoring_live",
                                       res, n, repetitions_requested=r, warmup=w,
                                       notes="Feeds epsilon_corr. The ONLY StatisticalSelfTester "
                                             "method actually called by run_self_tests()."))

        if n >= 1000:
            res = run_repeated(lambda: quantum_tester.dimension_witness(bits, bases), r, w,
                                cfg.timeout_seconds, sample_rss=(n >= 100_000))
            benchmarks.append(make_record("dimension_witness", "trust_monitoring_live",
                                           res, n, repetitions_requested=r, warmup=w,
                                           notes="Feeds epsilon_leak."))
        else:
            benchmarks.append(make_record(
                "dimension_witness", "trust_monitoring_live",
                {"status": "skipped", "error": None, "successful_runs": 0, "failed_runs": 0, "samples": None},
                n, repetitions_requested=r, warmup=w,
                notes="n<1000: dimension_witness short-circuits (returns (True,1.0) "
                      "without doing work) — not a real skip, just documented."))

        drift_monitor = PhysicalDriftMonitor()
        for _ in range(60):   # untimed warm-up past warmup_samples=50, see D_v16.py
            drift_monitor.update_efficiency(float(np.mean(bits)))

        def _cusum_step():
            drift_monitor.update_efficiency(float(np.mean(bits)))
            drift_monitor.detect_drift()

        res = run_repeated(_cusum_step, r, w, cfg.timeout_seconds, sample_rss=False)
        benchmarks.append(make_record("drift_cusum_steady_state", "trust_monitoring_live",
                                       res, n, repetitions_requested=r, warmup=w,
                                       notes="PhysicalDriftMonitor.update_efficiency+detect_drift, "
                                             "measured post-warmup (steady-state CUSUM cost). "
                                             "Feeds epsilon_drift."))

        tv = TrustVector(0.1, 0.1, 0.1, 0.1)
        res = run_repeated(lambda: tv.trust_score(), r, w, cfg.timeout_seconds, sample_rss=False)
        benchmarks.append(make_record("trust_score", "trust_monitoring_live",
                                       res, n, repetitions_requested=r, warmup=w,
                                       notes="TrustVector.trust_score(); O(1), included per spec."))

        probe = TrustEnhancedQRNG(block_size=n)
        res = run_repeated(lambda: probe.run_self_tests(bits, bases, raw_signal,
                                                          signal_stats=(0.0, 1.0)),
                            r, w, cfg.timeout_seconds, sample_rss=(n >= 100_000))
        benchmarks.append(make_record("run_self_tests_full", "trust_monitoring_live",
                                       res, n, repetitions_requested=r, warmup=w,
                                       notes="Whole live trust-vector-construction call, one shot."))

        for name, fn, cat, note in [
            ("santha_vazirani_test", lambda: stat_tester.santha_vazirani_test(bits),
             "trust_monitoring_live",
             "Called by run_self_tests(); feeds epsilon_corr (max-fused with the "
             "autocorrelation and runs-test signals)."),
            ("runs_test", lambda: stat_tester.runs_test(bits),
             "trust_monitoring_live",
             "Called by run_self_tests(); feeds epsilon_corr (max-fused with the "
             "autocorrelation and Santha-Vazirani signals)."),
            ("frequency_test", lambda: stat_tester.frequency_test(bits),
             "trust_monitoring_defined_not_called",
             "Defined in StatisticalSelfTester but not called by run_self_tests(); "
             "epsilon_bias uses an inline mean calculation. Not part of live cost."),
        ]:
            res = run_repeated(fn, r, w, cfg.timeout_seconds, sample_rss=(n >= 100_000))
            benchmarks.append(make_record(name, cat, res, n,
                                           repetitions_requested=r, warmup=w, notes=note))

        for rec in benchmarks[-9:]:
            if rec["block_size"] == n:
                if rec["status"] == "ok" and rec["latency_ms"]:
                    print(f"    [{rec['category']:26s}] {rec['component']:26s} "
                          f"{rec['latency_ms']['mean']:9.4f} ms  "
                          f"cpu_time={rec['cpu_time_ms']['mean']:7.4f} ms")
                elif rec["status"] == "skipped":
                    print(f"    [{rec['category']:26s}] {rec['component']:26s} SKIPPED "
                          f"({rec['error'] or rec['notes'][:50]})")


# ===========================================================================
# Section B — Security / certification-plane components
# ===========================================================================

def benchmark_certification(cfg: ExperimentConfig, benchmarks: List[Dict]) -> None:
    print("\n[B] Security-certification components "
          "(Hoeffding-corrected min-entropy certification + EAT + LHL length)")
    rng = np.random.RandomState(cfg.seed + 1)
    estimator = EntropyEstimator(security_parameter=1e-6)

    for n in cfg.cert_sizes:
        r, w = repetitions_for_size(n), warmup_for_size(n)
        bits = rng.randint(0, 2, size=n).astype(np.uint8)
        bases = rng.randint(0, 2, size=n).astype(np.uint8)
        print(f"  n={n:,}  (repeats={r}, warmup={w}) ...")

        res = run_repeated(lambda: estimator.certify_min_entropy(bits, bases), r, w,
                            cfg.timeout_seconds, sample_rss=False)
        rec = make_record("certify_min_entropy", "certification", res, n,
                           repetitions_requested=r, warmup=w,
                           notes="Hoeffding-corrected min-entropy certification. Hoeffding "
                                 "bound and min-entropy are computed inside ONE function in "
                                 "this codebase — not separable into two independently timed "
                                 "steps without modifying D_v16.py, which this benchmark does not do.")
        benchmarks.append(rec)
        if rec["status"] == "ok":
            print(f"    certify_min_entropy()  {rec['latency_ms']['mean']:9.4f} ms  "
                  f"cpu_time={rec['cpu_time_ms']['mean']:7.4f} ms")
        else:
            print(f"    certify_min_entropy()  SKIPPED ({rec['error']})")

        n_gen, h_min = n // 2, 0.9
        res = run_repeated(lambda: estimator.lhl_output_length(n_gen, h_min), r, w,
                            cfg.timeout_seconds, sample_rss=False)
        benchmarks.append(make_record("lhl_output_length", "certification", res, n,
                                       extra_param={"n_gen": n_gen, "h_min_certified": h_min},
                                       repetitions_requested=r, warmup=w,
                                       notes="Final certified output-length calculation. O(1)."))

    print("  EAT accumulation (QRNGSessionState.accumulate_eat) vs number of "
          "PRIOR BLOCKS t — this scales with block COUNT, not block SIZE "
          "(see architecture map, finding on accumulate_eat's O(t) list sums):")
    for t in cfg.eat_block_counts:
        session = QRNGSessionState()
        for _ in range(t):
            session.append_block(h_min_certified=0.85, n_gen=1_000_000)
        r = repetitions_for_size(t * 1000)
        res = run_repeated(lambda: session.accumulate_eat(epsilon_eat=5e-7), r, 1,
                            cfg.timeout_seconds, sample_rss=False)
        rec = make_record("accumulate_eat", "certification", res, block_size=None,
                           extra_param={"t_blocks": t}, repetitions_requested=r, warmup=1,
                           notes="Cost scales with t (number of accumulated blocks), "
                                 "not with block_size.")
        benchmarks.append(rec)
        if rec["status"] == "ok":
            print(f"    t={t:5d} blocks  {rec['latency_ms']['mean']:9.5f} ms")
        else:
            print(f"    t={t:5d} blocks  SKIPPED ({rec['error']})")


# ===========================================================================
# Section C — Extraction (FFT Toeplitz; dense Toeplitz documented absent)
# ===========================================================================

def benchmark_extraction(cfg: ExperimentConfig, benchmarks: List[Dict]) -> None:
    print("\n[C] Randomness extraction")
    print("  FFT-based Toeplitz extraction (the only extraction path present):")
    rng = np.random.RandomState(cfg.seed + 2)

    for n_gen, out_len in cfg.toeplitz_size_pairs:
        est_bytes = estimate_pipeline_memory_bytes(n_gen)
        safe, reason = check_memory_safety(est_bytes, cfg.safety_fraction)
        extra = {"n_gen": n_gen, "output_length": out_len}
        if not safe:
            benchmarks.append(make_record(
                "toeplitz_extract_fft", "extraction_fft",
                {"status": "skipped", "error": reason, "successful_runs": 0,
                 "failed_runs": 0, "samples": None}, block_size=n_gen, extra_param=extra))
            print(f"  n_gen={n_gen:,} out={out_len:,}  SKIPPED ({reason})")
            continue

        weak = rng.randint(0, 2, size=n_gen).astype(np.uint8)
        seed = rng.randint(0, 2, size=min(2 * out_len, 4096)).astype(np.uint8)
        extractor = RandomnessExtractor(input_length=n_gen, output_length=out_len)
        r, w = repetitions_for_size(n_gen), warmup_for_size(n_gen)

        res = run_repeated(lambda: extractor.toeplitz_extract(weak, seed), r, w,
                            cfg.timeout_seconds, sample_rss=True)
        throughput = None
        if res["status"] in ("ok", "partial"):
            mean_ms = stat_block(res["samples"]["latency_ms"])["mean"]
            throughput = (out_len / 1e6) / (mean_ms / 1000.0)
        rec = make_record("toeplitz_extract_fft", "extraction_fft", res, n_gen, extra,
                           repetitions_requested=r, warmup=w,
                           throughput_Mbits_per_s=throughput)
        benchmarks.append(rec)
        if rec["status"] in ("ok", "partial"):
            print(f"  n_gen={n_gen:>10,} out={out_len:>10,}  "
                  f"{rec['latency_ms']['mean']:10.3f} ms  "
                  f"cpu_time={rec['cpu_time_ms']['mean']:8.3f} ms  "
                  f"throughput={throughput:.3f} Mbit/s")
        else:
            print(f"  n_gen={n_gen:>10,} out={out_len:>10,}  SKIPPED ({rec['error']})")

    print("  Dense Toeplitz extraction — NOT PRESENT in this codebase.")
    benchmarks.append({
        "component": "dense_toeplitz_extraction", "category": "extraction_dense",
        "block_size": None, "extra_param": {},
        "repetitions_requested": 0, "warmup": 0, "successful_runs": 0, "failed_runs": 0,
        "status": "not_present_in_codebase", "error": None,
        "throughput_Mbits_per_s": None,
        "notes": ("RandomnessExtractor implements only the FFT-circulant path "
                  "(toeplitz_extract -> _toeplitz_fft_chunk, auto-chunked above "
                  "_MAX_CIRC_SIZE=2**23). No dense O(n*m) matrix-multiply Toeplitz "
                  "implementation exists in D_v16.py. Not benchmarked, not extrapolated "
                  "(spec sections 3 and 10). Theoretical complexity for reference only: "
                  "dense O(n*m) vs. FFT-circulant ~O(n log n); this is a textbook "
                  "complexity statement, not a measured or implied speedup number."),
        "latency_ms": None, "cpu_time_ms": None, "cpu_percent": None,
        "peak_python_tracked_MB": None, "peak_rss_sampled_MB": None, "rss_delta_MB": None,
    })

def benchmark_extraction_decomposition(cfg: ExperimentConfig, benchmarks: List[Dict]) -> None:
    """Splits extraction into seed expansion vs pure FFT convolution vs full call.
    Timing-only pass: tracemalloc and RSS sampler are OFF so they cannot distort
    Python-heavy code. Uses a pipeline-realistic seed of 2*out_len bits."""
    print("\n[C2] Extraction decomposition (seed expansion vs FFT vs full call)")
    rng = np.random.RandomState(cfg.seed + 9)
    for n_gen, out_len in cfg.toeplitz_size_pairs:
        if n_gen > 5_000_000:
            continue
        est = estimate_pipeline_memory_bytes(n_gen) + (n_gen + out_len) * 80
        safe, reason = check_memory_safety(est, cfg.safety_fraction)
        extra = {"n_gen": n_gen, "output_length": out_len}
        if not safe:
            benchmarks.append(make_record(
                "toeplitz_decomp_full", "extraction_decomposition",
                {"status": "skipped", "error": reason, "successful_runs": 0,
                 "failed_runs": 0, "samples": None}, n_gen, extra))
            print(f"  n_gen={n_gen:,}  SKIPPED ({reason})")
            continue

        weak = rng.randint(0, 2, size=n_gen).astype(np.uint8)
        seed = rng.randint(0, 2, size=2 * out_len).astype(np.uint8)
        required = n_gen + out_len - 1
        ext = RandomnessExtractor(input_length=n_gen, output_length=out_len)
        r, w = repetitions_for_size(n_gen), warmup_for_size(n_gen)
        r = max(r, 3)

        seed_full = ext._extend_seed(seed, required)   # untimed, reused by the FFT test

        specs = [
            ("toeplitz_decomp_seed_expansion", lambda: ext._extend_seed(seed, required),
             "SHA-256 seed expansion to n+m-1 bits (_extend_seed)."),
            ("toeplitz_decomp_fft_only", lambda: ext._toeplitz_fft_chunk(weak, seed_full, out_len),
             "Pure FFT circulant convolution (_toeplitz_fft_chunk), seed pre-expanded."),
            ("toeplitz_decomp_full", lambda: ext.toeplitz_extract(weak, seed),
             "Full toeplitz_extract() with a 2m-bit seed, as the pipeline calls it."),
        ]
        for name, fn, note in specs:
            res = run_repeated(fn, r, w, cfg.timeout_seconds,
                               sample_rss=False, track_mem=False)
            rec = make_record(name, "extraction_decomposition", res, n_gen, extra,
                              repetitions_requested=r, warmup=w, notes=note)
            benchmarks.append(rec)
            if rec["status"] in ("ok", "partial"):
                print(f"  n_gen={n_gen:>9,} {name:32s} {rec['latency_ms']['mean']:10.2f} ms "
                      f"(std {rec['latency_ms']['std']:.2f})")
            else:
                print(f"  n_gen={n_gen:>9,} {name:32s} SKIPPED ({rec['error']})")

# ===========================================================================
# Section D — process_block(): cold vs steady-state, plus true 4-method breakdown
# ===========================================================================

def benchmark_process_block(block_size: int, cfg: ExperimentConfig,
                             benchmarks: List[Dict]) -> Dict:
    print(f"\n[D] process_block() @ block_size={block_size:,} "
          f"(cold, steady-state, and internal breakdown)")

    est_bytes = estimate_pipeline_memory_bytes(block_size)
    safe, reason = check_memory_safety(est_bytes, cfg.safety_fraction)
    if not safe:
        print(f"  SKIPPED entirely ({reason})")
        for comp in ("process_block_total_cold", "process_block_total_steady",
                     "_certify_block", "_run_diagnostics", "_extract_block",
                     "_assemble_metadata"):
            benchmarks.append(make_record(
                comp, "end_to_end_cold" if "total_cold" in comp or comp.startswith("_") else "end_to_end_steady",
                {"status": "skipped", "error": reason, "successful_runs": 0,
                 "failed_runs": 0, "samples": None}, block_size))
        return {"status": "skipped", "reason": reason, "block_size": block_size}

    r, w = repetitions_for_size(block_size), warmup_for_size(block_size)

    # STEADY-state intentionally keeps ONE shared source instance (and one
    # shared TrustEnhancedQRNG instance) across all its repetitions — that
    # persistence is the entire point of the steady-state measurement
    # (PhysicalDriftMonitor's CUSUM state
    # are meant to accumulate). This is unchanged.
    steady_source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 3)

    # ---- COLD total: fresh TrustEnhancedQRNG + fresh QuantumSourceSimulator
    #      + fresh block per repetition. A shared source instance across
    #      "independent" cold trials would leak RNG-stream state between
    #      trials — harmless for the stateless IdealParams generator used
    #      here today, but it breaks the independence guarantee the cold
    #      condition is supposed to provide for any future/stateful source. ----
    def _build_cold(i):
        qrng = TrustEnhancedQRNG(block_size=block_size)
        fresh_source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 3 + i)
        block = fresh_source.generate_block(block_size)
        return (qrng, block)

    def _call_cold(item):
        qrng, block = item
        qrng.process_block(block.bits, block.bases, block.raw_signal)

    rec_cold = run_indexed_benchmark(
        "process_block_total_cold", "end_to_end_cold", block_size, w + r,
        _build_cold, _call_cold, r, w, cfg, benchmarks,
        notes="Fresh TrustEnhancedQRNG instance, fresh QuantumSourceSimulator, AND "
              "fresh block per repetition: measures fully independent single-block "
              "computational cost.",
        sample_rss=True)

    # ---- STEADY-state total: ONE instance, reused across repetitions ----
    steady_qrng = TrustEnhancedQRNG(block_size=block_size)
    steady_blocks = [steady_source.generate_block(block_size) for _ in range(w + r)]

    def _build_steady(i):
        return steady_blocks[i]

    def _call_steady(item):
        steady_qrng.process_block(item.bits, item.bases, item.raw_signal)

    rec_steady = run_indexed_benchmark(
        "process_block_total_steady", "end_to_end_steady", block_size, w + r,
        _build_steady, _call_steady, r, w, cfg, benchmarks,
        notes="ONE TrustEnhancedQRNG instance reused across all calls: measures "
              "operational steady-state cost (PhysicalDriftMonitor CUSUM state "
              "genuinely persists across calls by design).",
        sample_rss=True)

    # ---- 4-method breakdown (COLD; each step gets its own fresh fixtures) ----
    def build_certify(i):
        qrng = TrustEnhancedQRNG(block_size=block_size)
        fresh_source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 4000 + i)
        block = fresh_source.generate_block(block_size)
        session = QRNGSessionState()
        return (qrng, block, session)

    def call_certify(item):
        qrng, block, session = item
        qrng._certify_block(block.bits, block.bases, block.raw_signal, block_size, session)

    rec_certify = run_indexed_benchmark(
        "_certify_block", "end_to_end_breakdown", block_size, w + r,
        build_certify, call_certify, r, w, cfg, benchmarks,
            notes="Certification-plane step: BB84 split + Hoeffding/min-entropy "
              "certification + EAT history append, all inside one method in D_v16.py. "
              ,
        sample_rss=(block_size >= 100_000))

    def build_diag(i):
        qrng = TrustEnhancedQRNG(block_size=block_size)
        fresh_source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 5000 + i)
        block = fresh_source.generate_block(block_size)
        session = QRNGSessionState()
        c = qrng._certify_block(block.bits, block.bases, block.raw_signal, block_size, session)
        return (qrng, c)

    def call_diag(item):
        qrng, c = item
        try:
            qrng._run_diagnostics(c['raw_bits'], c['bases'], c['raw_signal'],
                                   (0.0, 1.0), c['h_min_certified'])
        except DiagnosticHaltError:
            pass

    rec_diag = run_indexed_benchmark(
        "_run_diagnostics", "end_to_end_breakdown", block_size, w + r,
        build_diag, call_diag, r, w, cfg, benchmarks,
        notes="The ONLY purely trust-monitoring step in the live block pipeline "
              "(calls run_self_tests() then applies halt/warn thresholds). "
              "_certify_block()'s output is precomputed UNTIMED before this step is timed.",
        sample_rss=(block_size >= 100_000))

    def build_extract(i):
        qrng = TrustEnhancedQRNG(block_size=block_size)
        fresh_source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 6000 + i)
        block = fresh_source.generate_block(block_size)
        session = QRNGSessionState()
        c = qrng._certify_block(block.bits, block.bases, block.raw_signal, block_size, session)
        return (qrng, c)

    def call_extract(item):
        qrng, c = item
        try:
            qrng._extract_block(c['gen_bits'], c['h_min_certified'], None)
        except InsufficientEntropyError:
            pass

    rec_extract = run_indexed_benchmark(
        "_extract_block", "end_to_end_breakdown", block_size, w + r,
        build_extract, call_extract, r, w, cfg, benchmarks,
        notes="PURE extraction step: LHL output-length calc + FFT Toeplitz extraction. "
              "_certify_block()'s output is precomputed UNTIMED before this step is timed.",
        sample_rss=(block_size >= 100_000))

    def build_assemble(i):
        qrng = TrustEnhancedQRNG(block_size=block_size)
        fresh_source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 7000 + i)
        block = fresh_source.generate_block(block_size)
        session = QRNGSessionState()
        c = qrng._certify_block(block.bits, block.bases, block.raw_signal, block_size, session)
        trust_vector, diagnostic_warning = qrng._run_diagnostics(
            c['raw_bits'], c['bases'], c['raw_signal'], (0.0, 1.0), c['h_min_certified'])
        output_length = qrng.entropy_estimator.lhl_output_length(c['n_gen'], c['h_min_certified'])
        extraction_rate = output_length / max(c['n_gen'], 1)
        try:
            output_bits, _ = qrng._extract_block(c['gen_bits'], c['h_min_certified'], None)
        except InsufficientEntropyError:
            output_bits = np.array([], dtype=np.uint8)
        cert_bundle = {'n_gen': c['n_gen'], 'n_test': c['n_test'],
                       'h_min_certified': c['h_min_certified'],
                       'output_length': output_length, 'extraction_rate': extraction_rate,
                       'cert': c['cert']}
        return (qrng, cert_bundle, c, trust_vector, diagnostic_warning, len(output_bits), session)

    def call_assemble(item):
        qrng, cert_bundle, c, trust_vector, diagnostic_warning, out_len, session = item
        qrng._assemble_metadata(cert_bundle, c['n_raw'], trust_vector,
                                 diagnostic_warning, out_len, session)

    rec_assemble = run_indexed_benchmark(
        "_assemble_metadata", "end_to_end_breakdown", block_size, w + r,
        build_assemble, call_assemble, r, w, cfg, benchmarks,
        notes="Metadata dict assembly + throughput-counter bookkeeping. All three "
              "prior steps precomputed UNTIMED before this step is timed.",
        sample_rss=False)

    breakdown = {"_certify_block": rec_certify, "_run_diagnostics": rec_diag,
                 "_extract_block": rec_extract, "_assemble_metadata": rec_assemble}

    if rec_cold["status"] != "ok" or not rec_cold.get("latency_ms"):
        print(f"  process_block_total_cold SKIPPED ({rec_cold.get('error')})")
        return {"status": "skipped", "reason": rec_cold.get("error"), "block_size": block_size,
                "cold": rec_cold, "steady": rec_steady, "breakdown": breakdown}

    total_ms = rec_cold["latency_ms"]["mean"]
    print(f"  process_block_total_cold    {total_ms:9.3f} ms  "
          f"cpu_time={rec_cold['cpu_time_ms']['mean']:8.3f} ms")
    if rec_steady["status"] == "ok" and rec_steady.get("latency_ms"):
        print(f"  process_block_total_steady  {rec_steady['latency_ms']['mean']:9.3f} ms  "
              f"cpu_time={rec_steady['cpu_time_ms']['mean']:8.3f} ms")

    parts_sum_ms = sum(v["latency_ms"]["mean"] for v in breakdown.values()
                        if v["status"] == "ok" and v.get("latency_ms"))
    overhead_ms = max(total_ms - parts_sum_ms, 0.0)
    ok_parts = [(k, v["latency_ms"]["mean"]) for k, v in breakdown.items()
                if v["status"] == "ok" and v.get("latency_ms")]
    bottleneck = max(ok_parts, key=lambda kv: kv[1], default=(None, 0.0))

    for name, rec in breakdown.items():
        if rec["status"] == "ok" and rec.get("latency_ms"):
            share = 100 * rec["latency_ms"]["mean"] / total_ms
            print(f"    {name:22s} {rec['latency_ms']['mean']:9.3f} ms  ({share:5.1f}% of cold total)")
        else:
            print(f"    {name:22s} SKIPPED ({rec.get('error')})")
    print(f"    {'[unaccounted overhead]':22s} {overhead_ms:9.3f} ms  "
          f"(dispatch/orchestration cost outside the 4 sub-methods)")
    if bottleneck[0]:
        print(f"    Bottleneck (RQ6): {bottleneck[0]} "
              f"({100*bottleneck[1]/total_ms:.1f}% of cold total)")

    est_output_bits = None
    if rec_assemble["status"] == "ok":
        pass  # exact output length available via breakdown fixtures only, not stored;
              # cold-total throughput below uses raw block_size / latency (input-side),
              # which is well-defined regardless.

    return {
        "status": "ok", "block_size": block_size,
        "cold": rec_cold, "steady": rec_steady, "breakdown": breakdown,
        "unaccounted_overhead_ms": overhead_ms,
        "bottleneck_component": bottleneck[0],
        "bottleneck_share_of_cold_total": (bottleneck[1] / total_ms) if total_ms else None,
        "throughput_raw_input_Mbits_per_s_cold": (block_size / 1e6) / (total_ms / 1000.0),
        "throughput_raw_input_Mbits_per_s_steady": (
            (block_size / 1e6) / (rec_steady["latency_ms"]["mean"] / 1000.0)
            if rec_steady["status"] == "ok" and rec_steady.get("latency_ms") else None),
    }


def benchmark_block_size_scaling(cfg: ExperimentConfig, benchmarks: List[Dict]) -> Dict:
    print("\n[E] Block-size scalability sweep (RQ5)")
    print(f"    Candidate sizes: {cfg.candidate_block_sizes}")
    print("    (10_000, 100_000, 1_000_000, 10_000_000 match this project's own "
          "convention seen in experiment_v2_1_v14.py / experiment_6_nist_validation_v3.py; "
          "3_240_000 is included as the project's stated actual published block size; "
          "3_500_000 is ALSO retained because it is the actual documented default "
          "(n_bits=3_500_000) used for Experiments 2-4 in experiment_v2_1_v14.py. "
          "Both are kept side by side, each with its own separately verifiable "
          "justification, rather than one silently replacing the other.)")
    results: Dict[str, Dict] = {}
    for size in cfg.candidate_block_sizes:
        results[str(size)] = benchmark_process_block(size, cfg, benchmarks)
    return results


# ===========================================================================
# Section F — Full generate_certified_random_bits(), REPEATED
# ===========================================================================

def benchmark_full_generation(cfg: ExperimentConfig, block_size: int,
                               benchmarks: List[Dict]) -> None:
    print(f"\n[F] Full generate_certified_random_bits() "
          f"(ideal source, block_size={block_size:,}, REPEATED per target)")

    for n_bits in cfg.full_gen_targets:
        # CertifiedGenerationSession.run() enforces block_size <= n_bits
        # (D_v16.py invariant, not something this harness may relax). Cap
        # per-target rather than crashing when a target is smaller than the
        # requested representative block_size.
        local_block_size = min(block_size, n_bits)
        if local_block_size != block_size:
            print(f"  n_bits={n_bits:,}: requested block_size={block_size:,} > n_bits; "
                  f"capping to block_size={local_block_size:,} for this target only.")

        approx_blocks = max(1, (n_bits // local_block_size) + 1)
        est_bytes = estimate_pipeline_memory_bytes(local_block_size) * approx_blocks
        safe, reason = check_memory_safety(est_bytes, cfg.safety_fraction)
        if not safe:
            benchmarks.append(make_record(
                "generate_certified_random_bits", "full_generation",
                {"status": "skipped", "error": reason, "successful_runs": 0,
                 "failed_runs": 0, "samples": None}, local_block_size,
                extra_param={"n_bits_requested": n_bits}))
            print(f"  n_bits={n_bits:,}  SKIPPED ({reason})")
            continue

        r = 3 if n_bits <= 200_000 else (2 if n_bits <= 2_000_000 else 1)
        w = 1 if r > 1 else 0

        def _build(i, _n_bits=n_bits, _block_size=local_block_size):
            source = QuantumSourceSimulator(IdealParams(), seed=cfg.seed + 100 + i)
            qrng = TrustEnhancedQRNG(block_size=_block_size)
            return (qrng, source, _n_bits)

        output_lengths: List[int] = []
        blocks_used_samples: List[int] = []

        def _call(item):
            qrng, source, n_target = item
            output_bits, metadata_list = qrng.generate_certified_random_bits(
                n_bits=n_target, source_simulator=source)
            # Record every successful call's actual output length. Never
            # overwrite with only the final repetition's value (fix #2) —
            # each of the warm-up call(s) and each timed repetition appends
            # its own entry here, in call order.
            output_lengths.append(len(output_bits))
            blocks_used_samples.append(
                metadata_list[-1].get("blocks_used", len(metadata_list) - 1))

        rec = run_indexed_benchmark(
            "generate_certified_random_bits", "full_generation", local_block_size, w + r,
            lambda i: _build(i), _call, r, w, cfg, benchmarks,
            extra_param={"n_bits_requested": n_bits},
            notes="Fresh QuantumSourceSimulator + fresh TrustEnhancedQRNG per repetition "
                  "(cold/independent trials). REPEATED (not single-shot) per the "
                  "measurement-methodology requirement.",
            sample_rss=True)

        if rec["status"] in ("ok", "partial") and rec.get("latency_ms"):
            n_success = rec["successful_runs"]
            # output_lengths also contains the untimed warm-up call(s), which
            # always run first (before the timed repetitions) and always
            # append before any timed call does. Slice off the trailing
            # n_success entries so only lengths from TIMED, successful
            # repetitions feed the throughput calculation below — this is
            # symmetric with how latency_ms/cpu_time_ms already exclude
            # warm-up in run_repeated().
            timed_lengths = output_lengths[-n_success:] if n_success > 0 else []
            mean_output_len = statistics.mean(timed_lengths) if timed_lengths else 0.0
            mean_s = rec["latency_ms"]["mean"] / 1000.0
            throughput = (mean_output_len / 1e6) / mean_s if mean_s > 0 else None
            rec["throughput_Mbits_per_s"] = throughput
            rec["extra_param"]["output_length_mean_actual"] = mean_output_len
            rec["extra_param"]["output_length_samples_used"] = len(timed_lengths)
            print(f"  n_bits={n_bits:>10,}  reps={rec['successful_runs']}/{w+r}  "
                  f"mean={rec['latency_ms']['mean']:.1f} ms  "
                  f"std={rec['latency_ms']['std']:.1f} ms  "
                  f"cpu_time={rec['cpu_time_ms']['mean']:.1f} ms  "
                  f"throughput={throughput:.4f} Mbit/s "
                  f"(mean output length={mean_output_len:.0f} bits over "
                  f"{len(timed_lengths)} timed repetitions: "
                  f"throughput = mean(output_lengths) / mean(latencies))")
        else:
            print(f"  n_bits={n_bits:>10,}  SKIPPED ({rec.get('error')})")


# ===========================================================================
# Overhead breakdown percentages (spec section 11)
# ===========================================================================

def compute_overhead_breakdown(scaling_results: Dict) -> Dict:
    rep_key = None
    if str(1_000_000) in scaling_results and scaling_results[str(1_000_000)].get("status") == "ok":
        rep_key = str(1_000_000)
    else:
        ok_keys = [k for k, v in scaling_results.items() if v.get("status") == "ok"]
        if ok_keys:
            rep_key = max(ok_keys, key=lambda k: int(k))
    if rep_key is None:
        return {"status": "unavailable", "reason": "no block-size sweep point completed"}

    r = scaling_results[rep_key]
    total_ms = r["cold"]["latency_ms"]["mean"]
    bd = r["breakdown"]

    def pct(name):
        rec = bd[name]
        if rec["status"] == "ok" and rec.get("latency_ms"):
            return 100 * rec["latency_ms"]["mean"] / total_ms
        return None

    return {
        "status": "ok",
        "representative_block_size": int(rep_key),
        "total_cold_ms": total_ms,
        "trust_monitoring_pct": pct("_run_diagnostics"),
        "certification_pct": pct("_certify_block"),
        "extraction_pct": pct("_extract_block"),
        "bookkeeping_pct": pct("_assemble_metadata"),
        "unaccounted_overhead_pct": 100 * r["unaccounted_overhead_ms"] / total_ms if total_ms else None,
                "caveat": ("'certification_pct' covers _certify_block(): the BB84 round split, "
                   "Hoeffding-bound min-entropy certification and the EAT history append "
                   "certification plane only;'trust_monitoring_pct' (from _run_diagnostics) is the "
                   "purely trust-monitoring share. This benchmark measures the existing "
                   "system and does not modify D_v16.py."),
    }


# ===========================================================================
# Validation checks (spec section 25)
# ===========================================================================

def validate_outputs(out: Path, benchmarks: List[Dict],
                      correctness_check: Dict, cfg: ExperimentConfig) -> Dict:
    checks = {}

    try:
        with open(out / "experiment_results.json") as f:
            data = json.load(f)
        checks["results_json_valid"] = True
    except Exception as exc:
        checks["results_json_valid"] = f"FAILED: {exc}"

    for csv_name in ("component_runtime.csv", "scalability.csv", "memory.csv",
                      "throughput.csv", "end_to_end.csv"):
        p = out / "tables" / csv_name
        try:
            with open(p, newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
            checks[f"csv_valid[{csv_name}]"] = f"OK ({len(rows)} data rows)"
        except Exception as exc:
            checks[f"csv_valid[{csv_name}]"] = f"FAILED: {exc}"

    for fig_name in ("latency_vs_block_size.png", "throughput_vs_block_size.png",
                      "memory_vs_block_size.png", "runtime_breakdown.png",
                      "toeplitz_comparison.png"):
        p = out / "figures" / fig_name
        if p.exists() and p.stat().st_size > 1000:
            checks[f"figure_exists[{fig_name}]"] = f"OK ({p.stat().st_size} bytes)"
        else:
            checks[f"figure_exists[{fig_name}]"] = "FAILED: missing or trivially small"

    for md_name in ("experiment_summary.md", "experiment_paper_section.md",
                     "reviewer_response_notes.md"):
        p = out / md_name
        checks[f"markdown_exists[{md_name}]"] = (
            "OK" if p.exists() and p.stat().st_size > 100 else "FAILED: missing or empty")

    nan_inf_found = []
    allowed_statuses = {"ok", "partial", "skipped", "not_present_in_codebase"}
    bad_statuses = []
    for rec in benchmarks:
        if rec["status"] not in allowed_statuses:
            bad_statuses.append((rec["component"], rec["status"]))
        for metric_key in ("latency_ms", "cpu_time_ms", "cpu_percent",
                            "peak_python_tracked_MB", "peak_rss_sampled_MB", "rss_delta_MB"):
            block = rec.get(metric_key)
            if block:
                for stat_name in ("mean", "median", "std", "min", "max"):
                    v = block.get(stat_name)
                    if v is not None and (isinstance(v, float) and (v != v or v in (float("inf"), float("-inf")))):
                        nan_inf_found.append((rec["component"], metric_key, stat_name))
    checks["no_nan_or_inf_in_reported_stats"] = "OK" if not nan_inf_found else f"FAILED: {nan_inf_found}"
    checks["all_statuses_from_allowed_set"] = "OK" if not bad_statuses else f"FAILED: {bad_statuses}"

    reported_sizes = {rec["block_size"] for rec in benchmarks
                       if rec["category"] in ("end_to_end_cold",) and rec["block_size"] is not None}
    expected_sizes = set(cfg.candidate_block_sizes)
    checks["block_sizes_match_config"] = (
        "OK" if reported_sizes.issubset(expected_sizes) or reported_sizes == expected_sizes
        else f"NOTE: reported={reported_sizes} vs configured={expected_sizes} "
             f"(mismatch expected if some sizes were skipped for safety)")

    checks["functional_correctness_toeplitz_extract"] = (
        "OK" if correctness_check.get("passed") else f"FAILED: {correctness_check}")

    print("\n[Validation] " + "=" * 60)
    for k, v in checks.items():
        print(f"  {k:50s} {v}")
    return checks


# ===========================================================================
# CSV table writers
# ===========================================================================

def _fmt(v, nd=4):
    return "" if v is None else (round(v, nd) if isinstance(v, float) else v)


def write_csv_component_runtime(benchmarks: List[Dict], out: Path) -> None:
    path = out / "tables" / "component_runtime.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "category", "block_size", "extra_param", "status",
                    "successful_runs", "failed_runs",
                    "latency_ms_mean", "latency_ms_median", "latency_ms_std",
                    "latency_ms_min", "latency_ms_max", "latency_ms_p95", "latency_ms_p99",
                    "cpu_time_ms_mean", "cpu_time_ms_std", "cpu_percent_mean", "notes"])
        for r in benchmarks:
            lm, ct, cp = r.get("latency_ms"), r.get("cpu_time_ms"), r.get("cpu_percent")
            w.writerow([
                r["component"], r["category"], _fmt(r["block_size"], 0),
                json.dumps(r.get("extra_param", {})), r["status"],
                r["successful_runs"], r["failed_runs"],
                _fmt(lm["mean"]) if lm else "", _fmt(lm["median"]) if lm else "",
                _fmt(lm["std"]) if lm else "", _fmt(lm["min"]) if lm else "",
                _fmt(lm["max"]) if lm else "", _fmt(lm["p95"]) if lm else "",
                _fmt(lm["p99"]) if lm else "",
                _fmt(ct["mean"]) if ct else "", _fmt(ct["std"]) if ct else "",
                _fmt(cp["mean"], 1) if cp else "", r.get("notes", ""),
            ])
    print(f"  Wrote: {path}")


def write_csv_scalability(scaling_results: Dict, out: Path) -> None:
    path = out / "tables" / "scalability.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_size", "status", "total_latency_ms_cold", "total_latency_ms_steady",
                    "certify_block_ms", "run_diagnostics_ms", "extract_block_ms",
                    "assemble_metadata_ms", "unaccounted_overhead_ms",
                    "bottleneck_component", "throughput_raw_input_Mbits_per_s_cold"])
        for size_str, r in scaling_results.items():
            if r.get("status") != "ok":
                w.writerow([size_str, r.get("status", "skipped")] + [""] * 9)
                continue
            bd = r["breakdown"]

            def gm(name):
                rec = bd[name]
                return _fmt(rec["latency_ms"]["mean"]) if rec["status"] == "ok" and rec.get("latency_ms") else ""

            w.writerow([
                size_str, "ok",
                _fmt(r["cold"]["latency_ms"]["mean"]),
                _fmt(r["steady"]["latency_ms"]["mean"]) if r["steady"]["status"] == "ok" and r["steady"].get("latency_ms") else "",
                gm("_certify_block"), gm("_run_diagnostics"), gm("_extract_block"),
                gm("_assemble_metadata"), _fmt(r["unaccounted_overhead_ms"]),
                r["bottleneck_component"] or "",
                _fmt(r["throughput_raw_input_Mbits_per_s_cold"]),
            ])
    print(f"  Wrote: {path}")


def write_csv_memory(benchmarks: List[Dict], out: Path) -> None:
    path = out / "tables" / "memory.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "category", "block_size", "status",
                    "peak_python_tracked_MB_mean", "peak_rss_sampled_MB_mean",
                    "rss_delta_MB_mean"])
        for r in benchmarks:
            ppy, prs, rd = (r.get("peak_python_tracked_MB"), r.get("peak_rss_sampled_MB"),
                            r.get("rss_delta_MB"))
            w.writerow([r["component"], r["category"], _fmt(r["block_size"], 0), r["status"],
                        _fmt(ppy["mean"]) if ppy else "", _fmt(prs["mean"]) if prs else "",
                        _fmt(rd["mean"]) if rd else ""])
    print(f"  Wrote: {path}")


def write_csv_throughput(benchmarks: List[Dict], out: Path) -> None:
    path = out / "tables" / "throughput.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "category", "block_size", "extra_param", "status",
                    "throughput_Mbits_per_s"])
        for r in benchmarks:
            if r.get("throughput_Mbits_per_s") is not None:
                w.writerow([r["component"], r["category"], _fmt(r["block_size"], 0),
                            json.dumps(r.get("extra_param", {})), r["status"],
                            _fmt(r["throughput_Mbits_per_s"])])
    print(f"  Wrote: {path}")


def write_csv_end_to_end(benchmarks: List[Dict], out: Path) -> None:
    path = out / "tables" / "end_to_end.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "category", "block_size_or_n_bits", "extra_param",
                     "status", "successful_runs", "failed_runs",
                     "latency_ms_mean", "latency_ms_std", "cpu_time_ms_mean",
                     "throughput_Mbits_per_s"])
        for r in benchmarks:
            if r["category"] in ("end_to_end_cold", "end_to_end_steady", "full_generation"):
                lm, ct = r.get("latency_ms"), r.get("cpu_time_ms")
                w.writerow([r["component"], r["category"], _fmt(r["block_size"], 0),
                            json.dumps(r.get("extra_param", {})), r["status"],
                            r["successful_runs"], r["failed_runs"],
                            _fmt(lm["mean"]) if lm else "", _fmt(lm["std"]) if lm else "",
                            _fmt(ct["mean"]) if ct else "",
                            _fmt(r.get("throughput_Mbits_per_s"))])
    print(f"  Wrote: {path}")


# ===========================================================================
# Figures (5, exactly named per spec section 14/16)
# ===========================================================================

def plot_latency_vs_block_size(scaling_results: Dict, out: Path) -> None:
    sizes, cold_ms, steady_ms = [], [], []
    for size_str, r in scaling_results.items():
        if r.get("status") == "ok" and r["cold"]["status"] == "ok":
            sizes.append(int(size_str))
            cold_ms.append(r["cold"]["latency_ms"]["mean"])
            steady_ms.append(r["steady"]["latency_ms"]["mean"]
                              if r["steady"]["status"] == "ok" and r["steady"].get("latency_ms") else None)
    fig, ax = plt.subplots(figsize=(9, 6))
    if sizes:
        order = np.argsort(sizes)
        sizes_arr = np.array(sizes)[order]
        cold_arr = np.array(cold_ms)[order]
        ax.plot(sizes_arr, cold_arr, "o-", label="process_block() — cold (fresh state)", linewidth=2)
        steady_arr = [steady_ms[i] for i in order]
        if all(v is not None for v in steady_arr):
            ax.plot(sizes_arr, steady_arr, "s--", label="process_block() — steady-state", linewidth=2)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Block size (bits)"); ax.set_ylabel("Execution time (ms, mean of repeated trials)")
    ax.set_title("Figure A — Latency vs Block Size (RQ2 / RQ5)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fpath = out / "figures" / "latency_vs_block_size.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_throughput_vs_block_size(scaling_results: Dict, out: Path) -> None:
    sizes, cold_tp, steady_tp = [], [], []
    for size_str, r in scaling_results.items():
        if r.get("status") == "ok":
            sizes.append(int(size_str))
            cold_tp.append(r["throughput_raw_input_Mbits_per_s_cold"])
            steady_tp.append(r["throughput_raw_input_Mbits_per_s_steady"])
    fig, ax = plt.subplots(figsize=(9, 6))
    if sizes:
        order = np.argsort(sizes)
        sizes_arr = np.array(sizes)[order]
        ax.plot(sizes_arr, np.array(cold_tp)[order], "^-", label="raw-input throughput — cold", linewidth=2)
        steady_arr = [steady_tp[i] for i in order]
        if all(v is not None for v in steady_arr):
            ax.plot(sizes_arr, steady_arr, "v--", label="raw-input throughput — steady-state", linewidth=2)
        ax.set_xscale("log")
    ax.set_xlabel("Block size (bits)"); ax.set_ylabel("Throughput (Mbit/s)")
    ax.set_title("Figure B — Throughput vs Block Size (RQ4 / RQ5)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fpath = out / "figures" / "throughput_vs_block_size.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_memory_vs_block_size(scaling_results: Dict, out: Path) -> None:
    sizes, rss_mb = [], []
    for size_str, r in scaling_results.items():
        if r.get("status") == "ok" and r["cold"]["status"] == "ok":
            prs = r["cold"].get("peak_rss_sampled_MB")
            if prs:
                sizes.append(int(size_str)); rss_mb.append(prs["mean"])
    fig, ax = plt.subplots(figsize=(9, 6))
    if sizes:
        order = np.argsort(sizes)
        ax.plot(np.array(sizes)[order], np.array(rss_mb)[order], "s-", linewidth=2)
        ax.set_xscale("log")
    ax.set_xlabel("Block size (bits)")
    ax.set_ylabel("Peak sampled RSS above baseline (MB, process_block cold)")
    ax.set_title("Figure C — Peak Memory vs Block Size (RQ3)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fpath = out / "figures" / "memory_vs_block_size.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_runtime_breakdown(overhead: Dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    if overhead.get("status") == "ok":
        labels = ["_run_diagnostics\n(trust monitoring,\nclean)",
                  "_certify_block\n(certification)",
                  "_extract_block\n(extraction,\nclean)",
                  "_assemble_metadata\n(bookkeeping)",
                  "unaccounted\noverhead"]
        vals = [overhead["trust_monitoring_pct"], overhead["certification_pct"],
                overhead["extraction_pct"], overhead["bookkeeping_pct"],
                overhead["unaccounted_overhead_pct"]]
        vals = [v if v is not None else 0.0 for v in vals]
        ax.bar(labels, vals)
        ax.set_ylabel("% of cold-mode process_block() total")
        ax.set_title(f"Figure D — Runtime Breakdown (RQ6)\n"
                     f"block_size={overhead['representative_block_size']:,}  |  "
                     f"total={overhead['total_cold_ms']:.2f} ms")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No successful block-size sweep point available",
                ha="center", va="center")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    fpath = out / "figures" / "runtime_breakdown.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_toeplitz_comparison(benchmarks: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    fft_recs = [r for r in benchmarks if r["component"] == "toeplitz_extract_fft"
                and r["status"] in ("ok", "partial")]
    if fft_recs:
        sizes = [r["block_size"] for r in fft_recs]
        lat = [r["latency_ms"]["mean"] for r in fft_recs]
        order = np.argsort(sizes)
        ax.plot(np.array(sizes)[order], np.array(lat)[order], "o-",
                label="FFT-based Toeplitz extraction (measured)", linewidth=2)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Input length n_gen (bits)"); ax.set_ylabel("Extraction latency (ms)")
    ax.set_title("Figure E — FFT vs. Dense Toeplitz\n"
                  "Dense Toeplitz: NOT PRESENT in this codebase — not measured, not extrapolated")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    ax.annotate("Dense Toeplitz implementation does not exist in D_v16.py.\n"
                "Textbook complexity only, for reference: dense O(n·m) vs. FFT ~O(n log n).\n"
                "No dense curve is plotted here because none was measured.",
                xy=(0.5, 0.02), xycoords="axes fraction", ha="center", fontsize=8,
                style="italic")
    plt.tight_layout()
    fpath = out / "figures" / "toeplitz_comparison.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ===========================================================================
# Markdown reports
# ===========================================================================

def write_experiment_summary_md(benchmarks: List[Dict], scaling_results: Dict,
                                 overhead: Dict, sysinfo: Dict, validation: Dict,
                                 correctness: Dict, timing_meta: Dict, out: Path) -> None:
    lines = []
    lines.append("# Experiment Summary\n")
    lines.append("Computational overhead, latency, memory, throughput, and scalability "
                 "evaluation of the existing TE-SI-QRNG pipeline. Measurement-only; no "
                 "architecture or security-model changes were made.\n")
    lines.append("## Environment\n")
    lines.append(f"- CPU: {sysinfo['cpu_model']} "
                 f"({sysinfo['logical_cpu_count']} logical / {sysinfo['physical_cpu_count']} physical)")
    lines.append(f"- RAM: {sysinfo['total_ram_GB']} GB total, "
                 f"{sysinfo['available_ram_GB_at_start']} GB available at run start")
    lines.append(f"- OS: {sysinfo['os']}")
    lines.append(f"- Python: {sysinfo['python_version']}  |  NumPy: {sysinfo['numpy_version']}  |  "
                 f"SciPy: {sysinfo['scipy_version']}")
    lines.append(f"- Measurement-phase wall time: {timing_meta['measurement_phase_s']:.1f} s  |  "
                 f"Reporting-phase (plots/CSV/MD) wall time: {timing_meta['reporting_phase_s']:.1f} s\n")
    lines.append("## RQ1/RQ6 — Component cost and bottleneck\n")
    if overhead.get("status") == "ok":
        lines.append(f"Representative block_size = {overhead['representative_block_size']:,} "
                     f"(cold-mode process_block() total = {overhead['total_cold_ms']:.2f} ms)\n")
        lines.append("| Stage | % of total | Notes |")
        lines.append("|---|---|---|")
        lines.append(f"| _run_diagnostics (trust monitoring) | "
                     f"{overhead['trust_monitoring_pct']:.1f}% | clean, trust-plane only |")
        lines.append(f"| _certify_block (certification) | "
                     f"{overhead['certification_pct']:.1f}% | certification plane only — see note below |")
        lines.append(f"| _extract_block (FFT Toeplitz extraction) | "
                     f"{overhead['extraction_pct']:.1f}% | clean, extraction only |")
        lines.append(f"| _assemble_metadata (bookkeeping) | "
                     f"{overhead['bookkeeping_pct']:.1f}% | |")
        lines.append(f"| unaccounted overhead | "
                     f"{overhead['unaccounted_overhead_pct']:.1f}% | dispatch/orchestration |\n")
        lines.append(f"> **Caveat:** {overhead['caveat']}\n")
    else:
        lines.append(f"Unavailable: {overhead.get('reason')}\n")
    lines.append("## RQ3/RQ5 — Scaling with block size\n")
    lines.append("| Block size | Status | Cold total (ms) | Steady total (ms) | "
                 "Peak sampled RSS (MB) | Raw-input throughput, cold (Mbit/s) |")
    lines.append("|---|---|---|---|---|---|")
    for size_str, r in scaling_results.items():
        if r.get("status") == "ok":
            prs = r["cold"].get("peak_rss_sampled_MB")
            steady_mean = (r["steady"]["latency_ms"]["mean"]
                           if r["steady"]["status"] == "ok" and r["steady"].get("latency_ms")
                           else None)
            steady_text = f"{steady_mean:.2f}" if steady_mean is not None else "n/a"
            rss_text = f"{prs['mean']:.2f}" if prs else "n/a"
            lines.append(f"| {int(size_str):,} | ok | {r['cold']['latency_ms']['mean']:.2f} | "
                         f"{steady_text} | {rss_text} | "
                         f"{r['throughput_raw_input_Mbits_per_s_cold']:.4f} |")
        else:
            lines.append(f"| {int(size_str):,} | skipped | — | — | — | — ({r.get('reason')}) |")
    lines.append("")
    lines.append("## Validation checks\n")
    lines.append("```")
    for k, v in validation.items():
        lines.append(f"{k}: {v}")
    lines.append("```\n")
    lines.append("## Functional correctness self-check\n")
    lines.append(f"```\n{json.dumps(correctness, indent=2)}\n```\n")
    lines.append("See `experiment_results.json` for the complete normalized record set, "
                 "`tables/*.csv` for tabular exports, and `figures/*.png` for plots.\n")

    with open(out / "experiment_summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote: {out / 'experiment_summary.md'}")


def write_experiment_paper_section_md(overhead: Dict, scaling_results: Dict,
                                       full_gen_records: List[Dict], sysinfo: Dict, out: Path) -> None:
    lines = []
    lines.append("# Computational Overhead and Latency of TE-SI-QRNG (Draft Section)\n")
    lines.append("## Experimental objective\n")
    lines.append("This experiment quantifies the computational cost, memory footprint, "
                 "throughput, and scaling behaviour of the implemented TE-SI-QRNG software "
                 "pipeline, addressing a reviewer request for computational overhead and "
                 "latency figures for the trust-monitoring tests and the FFT-based Toeplitz "
                 "extractor, particularly with respect to real-time deployment.\n")
    lines.append("## Methodology\n")
    lines.append("Each pipeline component was benchmarked individually using repeated timed "
                 "trials preceded by untimed warm-up calls, with wall-clock latency measured "
                 "via a monotonic high-resolution timer and CPU cost measured via per-process "
                 "CPU time. End-to-end block processing was measured in two modes: a "
                 "*cold* mode using a freshly constructed pipeline instance and block for "
                 "every trial (independent single-block cost), and a *steady-state* mode "
                 "reusing one pipeline instance across trials (operational cost, including "
                 "the drift monitor's persistent CUSUM state). The internal structure of the "
                 "block-processing routine was further decomposed into its four constituent "
                 "steps to attribute runtime to trust monitoring, certification, extraction, "
                 "and bookkeeping individually.\n")
    lines.append("## Environment\n")
    lines.append(f"All measurements were taken on {sysinfo['cpu_model']} "
                 f"({sysinfo['logical_cpu_count']} logical CPUs), {sysinfo['total_ram_GB']} GB "
                 f"RAM, Python {sysinfo['python_version']}, NumPy {sysinfo['numpy_version']}, "
                 f"SciPy {sysinfo['scipy_version']}, using the software-simulated quantum "
                 f"source described in New_simulator_v9.py (ideal-source configuration unless "
                 f"otherwise noted). No physical QRNG hardware was used.\n")
    lines.append("## Results\n")
    if overhead.get("status") == "ok":
        lines.append(f"At a representative block size of {overhead['representative_block_size']:,} "
                     f"bits, end-to-end block processing (cold mode) took "
                     f"{overhead['total_cold_ms']:.2f} ms on average. Of this, "
                     f"{overhead['trust_monitoring_pct']:.1f}% is attributable to the "
                     f"purely trust-monitoring diagnostic step, and "
                     f"{overhead['extraction_pct']:.1f}% to FFT-based Toeplitz extraction. "
                     f"A further {overhead['certification_pct']:.1f}% is "
                     f"attributable to the certification step (BB84 round split, "
                     f"Hoeffding-bound min-entropy certification and entropy-accumulation "
                     f"bookkeeping). See `runtime_breakdown.png` "
                     f"and `scalability.csv`/`throughput.csv` for the full data.\n")
    else:
        lines.append("[Insert measured numbers once a run has completed successfully; no "
                     "block-size sweep point completed in the run this draft was generated "
                     "from.]\n")
    lines.append("## Interpretation\n")
    lines.append("[Fill in after inspecting the generated figures: e.g., which component "
                 "dominates cost, how cost scales with block size, and whether cold vs. "
                 "steady-state measurements diverge meaningfully.]\n")
    lines.append("## Practical implications\n")
    lines.append("[Deployment-analysis / extrapolation only, not a hardware claim: compare "
                 "the measured certified-output throughput against a specific target "
                 "application's required bit rate before making any real-time feasibility "
                 "statement.]\n")
    lines.append("## Limitations\n")
    lines.append("- This is a software/simulation performance evaluation, not a physical "
                 "SI-QRNG hardware measurement.")
    lines.append("- NIST SP 800-22 statistical validation is a separate, pre-existing "
                 "experiment and is not re-evaluated here.")
    lines.append("- Passing statistical tests or achieving a given throughput is not a "
                 "cryptographic security claim.")
    lines.append("- Sampled process RSS is measured above a per-call baseline in one "
                 "long-lived process, so it can depend on allocator reuse from earlier, "
                 "larger block sizes; it is not a theoretical memory bound.")
    lines.append("- Above the single-FFT size limit the extractor switches to a chunked "
                 "path (the 10M-bit point), so latency is not strictly monotonic in "
                 "block size.")
    lines.append("- No dense Toeplitz implementation exists in the codebase to compare "
                 "against the FFT-based extractor.\n")

    with open(out / "experiment_paper_section.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote: {out / 'experiment_paper_section.md'}")


def write_reviewer_response_notes_md(out: Path) -> None:
    text = """# Reviewer-Response Notes

Addressing: "It would be useful to report computational overhead and latency of the
four monitoring tests and the FFT-based Toeplitz extraction, particularly if the
proposed architecture is intended for real-time deployment."

## Addressed by this experiment
- Computational overhead and latency of each trust-monitoring component actually
  invoked by the live pipeline (bias, drift/CUSUM, autocorrelation, Santha-Vazirani, runs, leakage,
  trust-vector construction, trust-score computation).
- Computational overhead and latency of the security-certification components
  (Hoeffding-corrected min-entropy certification, EAT accumulation, final
  certified-output-length calculation).
- Computational overhead and latency of the FFT-based Toeplitz extractor across
  a range of input/output sizes, including the automatic chunked-extraction
  code path.
- Peak memory (Python-tracked and sampled process RSS) for each component and
  for full block processing.
- End-to-end block-processing latency and throughput, in both cold (independent
  single-block) and steady-state (operational, persistent-state) modes.
- Scaling behaviour with block size, from 10^4 to 10^7 bits (where memory-safe
  on the measurement machine).
- A software-deployment-oriented discussion of the measured throughput, framed
  explicitly as deployment analysis against a stated target rate.

## Not addressed by this experiment
- Physical SI-QRNG hardware validation.
- Optical interference, detector noise, or SPAD dead-time effects.
- Physical side-channel injection or laboratory environmental drift.
- Any claim that NIST SP 800-22 compliance constitutes a proof of quantum
  randomness, min-entropy, or composable/cryptographic security.
- Any claim that the system has been validated on physical hardware.

These remain future experimental work, contingent on access to physical hardware,
and are explicitly out of scope for this computational-overhead evaluation.
"""
    with open(out / "reviewer_response_notes.md", "w") as f:
        f.write(text)
    print(f"  Wrote: {out / 'reviewer_response_notes.md'}")


# ===========================================================================
# Entry point
# ===========================================================================

def run_all(cfg: ExperimentConfig) -> Dict:
    out = Path(cfg.output_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    sysinfo = get_system_info()
    print("=" * 80)
    print("EXPERIMENT: Computational Overhead, Latency, Memory, Throughput, "
          "and Scalability of TE-SI-QRNG")
    print("=" * 80)
    print(f"CPU: {sysinfo['cpu_model']}")
    print(f"{sysinfo['logical_cpu_count']} logical / {sysinfo['physical_cpu_count']} physical CPUs, "
          f"{sysinfo['total_ram_GB']} GB RAM total, "
          f"{sysinfo['available_ram_GB_at_start']} GB available now")
    print(f"OS: {sysinfo['os']}  |  Python {sysinfo['python_version']}  |  "
          f"NumPy {sysinfo['numpy_version']}  |  SciPy {sysinfo['scipy_version']}")
    print(f"Safety fraction: {cfg.safety_fraction}  |  Timeout: {cfg.timeout_seconds}s  |  "
          f"POSIX timeout guard active: {sysinfo['posix_timeout_guard_active']}")

    if cfg.quick:
        cfg.self_test_sizes = [10_000, 100_000]
        cfg.cert_sizes = [10_000, 100_000]
        cfg.eat_block_counts = [1, 10, 50]
        cfg.toeplitz_size_pairs = [(10_000, 5_000), (100_000, 50_000)]
        cfg.candidate_block_sizes = [10_000, 100_000]
        cfg.full_gen_targets = [50_000]

    with open(out / "experiment_config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)
    print(f"\n  Wrote: {out / 'experiment_config.json'}")

    t_measure_start = time.time()

    benchmarks: List[Dict] = []
    correctness = functional_correctness_check(benchmarks)
    benchmark_trust_monitoring(cfg, benchmarks)
    benchmark_certification(cfg, benchmarks)
    benchmark_extraction(cfg, benchmarks)
    benchmark_extraction_decomposition(cfg, benchmarks)
    scaling_results = benchmark_block_size_scaling(cfg, benchmarks)
    rep_block_size = min(cfg.candidate_block_sizes[-1], 1_000_000)
    benchmark_full_generation(cfg, rep_block_size, benchmarks)

    t_measure_end = time.time()
    measurement_phase_s = t_measure_end - t_measure_start

    overhead = compute_overhead_breakdown(scaling_results)

    t_report_start = time.time()

    results_json = {
        "experiment": ("Computational overhead, latency, memory, throughput, and "
                       "scalability evaluation of the existing TE-SI-QRNG pipeline "
                       "(measurement-only; no architecture or security-model changes)"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": sysinfo,
        "software": {"python": sysinfo["python_version"], "numpy": sysinfo["numpy_version"],
                     "scipy": sysinfo["scipy_version"], "psutil": sysinfo["psutil_version"],
                     "matplotlib": sysinfo["matplotlib_version"]},
        "configuration": cfg.to_dict(),
        "benchmarks": benchmarks,
        "block_size_scaling": scaling_results,
        "overhead_breakdown": overhead,
        "functional_correctness_check": correctness,
        "timing_meta": {"measurement_phase_s": measurement_phase_s},
    }
    with open(out / "experiment_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Wrote: {out / 'experiment_results.json'}")

    write_csv_component_runtime(benchmarks, out)
    write_csv_scalability(scaling_results, out)
    write_csv_memory(benchmarks, out)
    write_csv_throughput(benchmarks, out)
    write_csv_end_to_end(benchmarks, out)

    plot_latency_vs_block_size(scaling_results, out)
    plot_throughput_vs_block_size(scaling_results, out)
    plot_memory_vs_block_size(scaling_results, out)
    plot_runtime_breakdown(overhead, out)
    plot_toeplitz_comparison(benchmarks, out)

    t_report_end = time.time()
    timing_meta = {"measurement_phase_s": measurement_phase_s,
                    "reporting_phase_s": t_report_end - t_report_start}

    validation = validate_outputs(out, benchmarks, correctness, cfg)
    write_experiment_summary_md(benchmarks, scaling_results, overhead, sysinfo,
                                 validation, correctness, timing_meta, out)
    write_experiment_paper_section_md(overhead, scaling_results, benchmarks, sysinfo, out)
    write_reviewer_response_notes_md(out)
    # Re-validate now that all three Markdown files exist, then rewrite the
    # summary once so it records the true final validation result.
    validation = validate_outputs(out, benchmarks, correctness, cfg)
    write_experiment_summary_md(benchmarks, scaling_results, overhead, sysinfo,
                                 validation, correctness, timing_meta, out)

    results_json["timing_meta"]["reporting_phase_s"] = timing_meta["reporting_phase_s"]
    results_json["validation"] = validation
    with open(out / "experiment_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nMeasurement phase: {measurement_phase_s:.1f}s  |  "
          f"Reporting phase: {timing_meta['reporting_phase_s']:.1f}s")
    print(f"\nAll outputs written under: {out.resolve()}")
    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--safety-fraction", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()

    timeout = None if args.timeout_seconds == 0 else args.timeout_seconds
    
    cfg = ExperimentConfig(output_dir=args.output_dir, quick=args.quick,
                            safety_fraction=args.safety_fraction,
                            timeout_seconds=timeout, seed=args.seed)
    run_all(cfg)