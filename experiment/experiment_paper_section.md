# Computational Overhead and Latency of TE-SI-QRNG (Draft Section)

## Experimental objective

This experiment quantifies the computational cost, memory footprint, throughput, and scaling behaviour of the implemented TE-SI-QRNG software pipeline, addressing a reviewer request for computational overhead and latency figures for the trust-monitoring tests and the FFT-based Toeplitz extractor, particularly with respect to real-time deployment.

## Methodology

Each pipeline component was benchmarked individually using repeated timed trials preceded by untimed warm-up calls, with wall-clock latency measured via a monotonic high-resolution timer and CPU cost measured via per-process CPU time. End-to-end block processing was measured in two modes: a *cold* mode using a freshly constructed pipeline instance and block for every trial (independent single-block cost), and a *steady-state* mode reusing one pipeline instance across trials (operational cost, including the drift monitor's persistent CUSUM state). The internal structure of the block-processing routine was further decomposed into its four constituent steps to attribute runtime to trust monitoring, certification, extraction, and bookkeeping individually.

## Environment

All measurements were taken on AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD (12 logical CPUs), 7.33 GB RAM, Python 3.14.3, NumPy 2.4.3, SciPy 1.17.1, using the software-simulated quantum source described in New_simulator_v9.py (ideal-source configuration unless otherwise noted). No physical QRNG hardware was used.

## Results

At a representative block size of 1,000,000 bits, end-to-end block processing (cold mode) took 3765.10 ms on average. Of this, 15.2% is attributable to the purely trust-monitoring diagnostic step, and 83.7% to FFT-based Toeplitz extraction. A further 1.2% is attributable to a combined step performing pre-value gating together with Hoeffding-bound min-entropy certification and entropy-accumulation bookkeeping, which the current implementation does not separate into independently measurable units; this combined figure should not be described as "certification cost" alone. See `runtime_breakdown.png` and `scalability.csv`/`throughput.csv` for the full data.

## Interpretation

[Fill in after inspecting the generated figures: e.g., which component dominates cost, how cost scales with block size, and whether cold vs. steady-state measurements diverge meaningfully.]

## Practical implications

[Deployment-analysis / extrapolation only, not a hardware claim: compare the measured certified-output throughput against a specific target application's required bit rate before making any real-time feasibility statement.]

## Limitations

- This is a software/simulation performance evaluation, not a physical SI-QRNG hardware measurement.
- NIST SP 800-22 statistical validation is a separate, pre-existing experiment and is not re-evaluated here.
- Passing statistical tests or achieving a given throughput is not a cryptographic security claim.
- The gating+certification runtime share could not be cleanly separated without modifying the measured implementation (see Results).
- No dense Toeplitz implementation exists in the codebase to compare against the FFT-based extractor.
