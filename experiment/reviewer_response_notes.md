# Reviewer-Response Notes

Addressing: "It would be useful to report computational overhead and latency of the
four monitoring tests and the FFT-based Toeplitz extraction, particularly if the
proposed architecture is intended for real-time deployment."

## Addressed by this experiment
- Computational overhead and latency of each trust-monitoring component actually
  invoked by the live pipeline (bias, drift/CUSUM, autocorrelation, leakage,
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
