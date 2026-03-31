"""Global config for TE-SI-QRNG experiments."""

DEFAULT_N_BITS = 10_000_000
DEFAULT_BLOCK_SIZE = 3_000_000

if DEFAULT_BLOCK_SIZE > DEFAULT_N_BITS:
    raise ValueError("block_size must be <= n_bits")
