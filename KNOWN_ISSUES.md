# Known issues in the published-results code

This note applies to the code that produced the results in the article "Trust-enhanced source-independent quantum random number generation with continuous per-block self-testing" (Quantum Information Processing, doi:10.1007/s11128-026-05388-7).

- **Published-results code:** commit `35e40ca` on branch `fix/extractor-float64-random-excursions`. The code on that branch is unchanged, and only this note was added.
- **Corrected code:** branch `fix/runs-test-and-toeplitz-seed`. It contains the two fixes below. It will not reproduce the trust scores in the article.

## 1. The runs test always returns p = 1

`StatisticalSelfTester.runs_test` in `D_v16.py` computed the variance with the term `2np(1-p) - n`. The correct term is `2np(1-p) - 1`. The result was always negative, so the function returned p = 1.0 for every input.

- **Effect on the article:** the third statistic of Eq. (5) stayed at its floor, sigma(0; 10, 0.30) = 0.047, in all reported results.
- **Not affected:** the certified min-entropy bound and the output length, because the trust vector has no path to them.
- **Effect of the fix:** trust scores change and the sigmoid for this term needs recalibrating. Measured on 8 blocks per source (2 x 10^6 raw bits, test subset):

| Source | Mean T as published | Mean T with the fix |
|---|---|---|
| Ideal | 0.960 | 0.566 (range 0.502-0.924) |
| Phase noise (clean) | 0.960 | 0.657 |
| Small bias | 0.932 | 0.640 |
| Short correlation | 0.954 | 0.500 |
| Strong attack | 0.778 | 0.492 |
| Long correlation | 0.499 | 0.499 |

The corrected branch fixes the formula but does not recalibrate the sigmoid.

## 2. The Toeplitz matrix is symmetric

`RandomnessExtractor._toeplitz_fft_chunk` took the first column from `seed[:m]` and the first row from `seed[1:n]`, so both used the same bits. The matrix applied was T[i,j] = s[|i-j|], and only max(n, m) of the n + m - 1 seed bits were used.

- **Effect on the article:** this symmetric family is not two-universal, so Theorem 1 does not apply to it directly. If the input is palindromic about an output position, the output is palindromic too, and half of it is determined.
- **For i.i.d. sources, as Theorem 2 assumes:** the effect measured was small. Over sampled difference vectors at n = 256 and n = 1024, the mean collision penalty was 0 bits at m/n up to 0.94 for balanced bits, and about 3 bits at m/n = 0.99 in one of the two sizes tested. This is an empirical estimate and not a proof.
- **Not affected:** the certified min-entropy bound and the output length.
- **Effect of the fix:** extracted bits change. The extraction seed is unseeded, so the extracted output, and therefore the post-extraction NIST results, differ from run to run in any case.

The corrected branch takes the first row from `seed[m:m + n - 1]`. Its FFT product was checked against direct multiplication by the Toeplitz matrix.
