# Experiment Summary

Computational overhead, latency, memory, throughput, and scalability evaluation of the existing TE-SI-QRNG pipeline. Measurement-only; no architecture or security-model changes were made.

## Environment

- CPU: AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD (12 logical / 6 physical)
- RAM: 7.33 GB total, 3.56 GB available at run start
- OS: Windows-11-10.0.26200-SP0
- Python: 3.14.3  |  NumPy: 2.4.3  |  SciPy: 1.17.1
- Measurement-phase wall time: 627.0 s  |  Reporting-phase (plots/CSV/MD) wall time: 4.9 s

## RQ1/RQ6 — Component cost and bottleneck

Representative block_size = 1,000,000 (cold-mode process_block() total = 3765.10 ms)

| Stage | % of total | Notes |
|---|---|---|
| _run_diagnostics (trust monitoring) | 15.2% | clean, trust-plane only |
| _certify_block (gating + certification) | 1.2% | **MIXED** — see caveat below |
| _extract_block (FFT Toeplitz extraction) | 83.7% | clean, extraction only |
| _assemble_metadata (bookkeeping) | 0.0% | |
| unaccounted overhead | 0.0% | dispatch/orchestration |

> **Caveat:** 'gating_and_certification_mixed_pct' is NOT a clean 'certification plane only' number: _certify_block() in D_v16.py combines Layer-1 pre-value gating (trust/health-adjacent) with Hoeffding bound, min-entropy certification, and EAT history bookkeeping (security plane) inside a single method. This benchmark does not modify D_v16.py to split them further, per the instruction to measure the existing system rather than redesign it. 'trust_monitoring_pct' (from _run_diagnostics) IS a clean, purely-trust-monitoring number.

## RQ3/RQ5 — Scaling with block size

| Block size | Status | Cold total (ms) | Steady total (ms) | Peak sampled RSS (MB) | Raw-input throughput, cold (Mbit/s) |
|---|---|---|---|---|---|
| 10,000 | ok | 31.83 | 31.36 | 0.17 | 0.3142 |
| 100,000 | ok | 308.82 | 310.16 | 16.60 | 0.3238 |
| 1,000,000 | ok | 3765.10 | 3835.26 | 142.09 | 0.2656 |
| 3,240,000 | ok | 16931.78 | 16942.40 | 452.04 | 0.1914 |
| 3,500,000 | ok | 18630.75 | 18545.65 | 366.02 | 0.1879 |
| 10,000,000 | ok | 11644.34 | 11597.03 | 1361.11 | 0.8588 |

## Validation checks

```
results_json_valid: True
csv_valid[component_runtime.csv]: OK (82 data rows)
csv_valid[scalability.csv]: OK (6 data rows)
csv_valid[memory.csv]: OK (82 data rows)
csv_valid[throughput.csv]: OK (6 data rows)
csv_valid[end_to_end.csv]: OK (14 data rows)
figure_exists[latency_vs_block_size.png]: OK (182740 bytes)
figure_exists[throughput_vs_block_size.png]: OK (156758 bytes)
figure_exists[memory_vs_block_size.png]: OK (140024 bytes)
figure_exists[runtime_breakdown.png]: OK (137934 bytes)
figure_exists[toeplitz_comparison.png]: OK (205249 bytes)
markdown_exists[experiment_summary.md]: FAILED: missing or empty
markdown_exists[experiment_paper_section.md]: FAILED: missing or empty
markdown_exists[reviewer_response_notes.md]: FAILED: missing or empty
no_nan_or_inf_in_reported_stats: OK
all_statuses_from_allowed_set: OK
block_sizes_match_config: OK
functional_correctness_toeplitz_extract: OK
```

## Functional correctness self-check

```
{
  "check": "toeplitz_extract_functional_correctness",
  "passed": true,
  "output_length_expected": 8000,
  "output_length_actual": 8000,
  "output_dtype": "uint8",
  "error": null
}
```

See `experiment_results.json` for the complete normalized record set, `tables/*.csv` for tabular exports, and `figures/*.png` for plots.
