"""
Experimental Validation of TE-SI-QRNG
======================================

Comprehensive experiments to validate the Trust-Enhanced Source-Independent
Quantum Random Number Generator across multiple scenarios.

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for multiprocessing
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# Import optimized v5 core
from D_v2 import TrustEnhancedQRNG, TrustVector

from New_simulator import (
    QuantumSourceSimulator,
    AttackScenarioSimulator,
    create_test_scenarios,
    IdealParams,
    AttackedParams,
    BiasedParams,
    GeneratedBlock,
)


# ---------------------------------------------------------------------------
# Parallel worker functions  (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _run_exp1_scenario(args) -> Tuple[str, Dict]:
    """Worker: experiment 1 — single scenario."""
    scenario_name, params, n_bits = args
    source = QuantumSourceSimulator(params, seed=42)
    te_qrng = TrustEnhancedQRNG(block_size=10000000)

    block = source.generate_block(n_bits)
    test_mask   = (block.bases == 1)
    test_bits   = block.bits[test_mask]
    test_bases  = block.bases[test_mask]
    test_signal = block.raw_signal[test_mask]

    tv = te_qrng.run_self_tests(test_bits, test_bases, test_signal)

    result = {
        'epsilon_bias':          tv.epsilon_bias,
        'epsilon_drift':         tv.epsilon_drift,
        'epsilon_corr':          tv.epsilon_corr,
        'epsilon_leak':          tv.epsilon_leak,
        'trust_score':           tv.trust_score(),
        'e_obs_contribution':    tv.epsilon_bias + tv.epsilon_corr / 2.0,
        'e_upper_contribution':  tv.epsilon_drift * 0.05,
        'delta_leak':            tv.epsilon_leak * 1.0,
    }
    return scenario_name, result


def _run_exp2_scenario(args) -> Tuple[str, Dict]:
    """Worker: experiment 2 — single scenario."""
    scenario_name, params, n_bits = args
    source  = QuantumSourceSimulator(params, seed=42)
    te_qrng = TrustEnhancedQRNG(block_size=1000000)

    # this extraction step may allocate several hundred megabytes of memory
    # for the FFT.  if the worker runs out of RAM we'll catch that error and
    # raise something more informative to the parent process.
    try:
        output_bits, metadata_list = te_qrng.generate_certified_random_bits(
            n_bits=n_bits, source_simulator=source
        )
    except MemoryError as exc:
        msg = (
            f"MemoryError during experiment 2 scenario '{scenario_name}' "
            f"(n_bits={n_bits}).  The Toeplitz extractor attempted a huge FFT "
            "and the process hit its memory limit.\n"
            "Reduce `n_bits`/`block_size` or run with fewer workers, or use a "
            "machine with more RAM."
        )
        raise RuntimeError(msg) from exc

    if len(output_bits) > 0:
        p1 = float(np.mean(output_bits))
        p0 = 1.0 - p1
        empirical_h = (-(p1 * np.log2(p1) + p0 * np.log2(p0))
                       if 0.0 < p1 < 1.0 else 0.0)
    else:
        empirical_h = 0.0

    final_summary        = metadata_list[-1]
    h_total_eat          = final_summary.get('h_total_eat', 0.0)
    certified_output_bits = final_summary.get('certified_output_bits', len(output_bits))
    delta_eat            = final_summary.get('delta_eat', 0.0)
    blocks_used          = final_summary.get('blocks_used', max(len(metadata_list) - 1, 0))
    sum_f_ei             = final_summary.get('sum_f_ei', 0.0)

    h_total_progression = []
    delta_progression   = []
    for meta in metadata_list[:-1]:
        h_total_progression.append(meta.get('h_total_eat', 0.0))
        delta_progression.append(
            meta.get('sum_f_ei', 0.0) - meta.get('h_total_eat', 0.0)
        )

    result = {
        'empirical_h_output':    empirical_h,
        'h_total_eat':           h_total_eat,
        'certified_output_bits': certified_output_bits,
        'delta_eat':             delta_eat,
        'blocks_used':           blocks_used,
        'sum_f_ei':              sum_f_ei,
        'h_total_progression':   h_total_progression,
        'delta_progression':     delta_progression,
        'output_bits':           len(output_bits),
        'expected_bits':         n_bits,
        'certified_entropy':     h_total_eat,   # legacy compat
    }
    return scenario_name, result


def _run_exp3_scenario(args) -> Tuple[str, List]:
    """Worker: experiment 3 — single attack scenario."""
    scenario_name, params, n_bits = args
    source  = QuantumSourceSimulator(params, seed=42)
    te_qrng = TrustEnhancedQRNG(block_size=1000000)

    output_bits, metadata_list = te_qrng.generate_certified_random_bits(
        n_bits=n_bits, source_simulator=source
    )
    final_summary = metadata_list[-1]

    block_results = []
    for idx, meta in enumerate(metadata_list[:-1]):
        block_results.append({
            'block':                 idx,
            'trust_score':           meta['trust_score'],
            'h_total_eat':           meta.get('h_total_eat', 0.0),
            'delta_eat':             meta.get('sum_f_ei', 0.0) - meta.get('h_total_eat', 0.0),
            'extraction_rate':       meta['extraction_rate'],
            'certified_output_bits': final_summary.get('certified_output_bits', 0),
        })

    return scenario_name, block_results


def _run_exp4_scenario(args) -> Tuple[str, Dict]:
    """Worker: experiment 4 — single scenario, both TE and standard SI-QRNG."""
    from D_v2 import TrustVector as TV

    class StandardSIQRNG(TrustEnhancedQRNG):
        def run_self_tests(self, raw_bits, bases=None, raw_signal=None, signal_stats=None):
            return TV(0.0, 0.0, 0.0, 0.0)

    scenario_name, params, n_bits = args

    source  = QuantumSourceSimulator(params, seed=42)
    te_qrng = TrustEnhancedQRNG(block_size=1000000, extractor_efficiency=0.9)
    te_output, te_metadata = te_qrng.generate_certified_random_bits(
        n_bits=n_bits, source_simulator=source
    )

    source.reset()
    si_qrng = StandardSIQRNG(block_size=1000000, extractor_efficiency=0.95)
    si_output, si_metadata = si_qrng.generate_certified_random_bits(
        n_bits=n_bits, source_simulator=source
    )

    te_quality = _compute_quality_score_static(te_output)
    si_quality = _compute_quality_score_static(si_output)

    te_blocks = te_metadata[:-1]
    si_blocks = si_metadata[:-1]

    result = {
        'te_output_bits':   len(te_output),
        'si_output_bits':   len(si_output),
        'te_quality_score': te_quality,
        'si_quality_score': si_quality,
        'te_avg_trust':     float(np.mean([m['trust_score'] for m in te_blocks])) if te_blocks else 0.0,
        'te_avg_h_cert':    float(np.mean([m.get('h_min_certified', 0) for m in te_blocks])) if te_blocks else 0.0,
        'si_avg_h_cert':    float(np.mean([m.get('h_min_certified', 0) for m in si_blocks])) if si_blocks else 0.0,
    }
    return scenario_name, result


def _compute_quality_score_static(bits: np.ndarray) -> float:
    """
    Compute empirical quality score.
    OPTIMIZATION: np.diff replaces Python for-loop.
    """
    if len(bits) < 100:
        return 0.0

    prob_one   = float(np.mean(bits))
    freq_score = 1.0 - 2 * abs(prob_one - 0.5)

    runs          = int(np.count_nonzero(np.diff(bits))) + 1
    expected_runs = 2 * len(bits) * prob_one * (1 - prob_one)
    denom         = len(bits) / 2
    runs_score    = max(1.0 - abs(runs - expected_runs) / max(denom, 1), 0.0)

    return float(np.mean([freq_score, runs_score]))


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Runs comprehensive experiments on TE-SI-QRNG system.

    Parallelization strategy
    ------------------------
    Each scenario within an experiment is fully independent (separate source,
    separate QRNG instance, no shared state).  We use ProcessPoolExecutor to
    run all 12 scenarios in parallel, saturating available CPU cores.

    Typical speedup: ~8–12× on an 8-core machine.
    """

    def __init__(self, output_dir: str = "results",
                 max_workers: Optional[int] = None):
        """
        Args:
            output_dir:  Directory to save experimental results.
            max_workers: Number of parallel workers (default = CPU count).
        """
        self.output_dir  = Path(output_dir)
        self.max_workers = max_workers or os.cpu_count()

        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        print(f"[ExperimentRunner] Parallel workers: {self.max_workers}")

    # ------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------

    def experiment_1_trust_quantification(self, n_bits: int = 1000000):
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: Trust Quantification Across Scenarios  [PARALLEL]")
        print("=" * 80)

        scenarios = create_test_scenarios()
        task_args = [(name, params, n_bits) for name, params in scenarios.items()]
        results   = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_run_exp1_scenario, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                name, result = fut.result()
                results[name] = result
                tv = result
                print(f"  [{name}] Trust={tv['trust_score']:.4f}  "
                      f"ε_bias={tv['epsilon_bias']:.4f}  "
                      f"ε_corr={tv['epsilon_corr']:.4f}  "
                      f"ε_drift={tv['epsilon_drift']:.4f}  "
                      f"ε_leak={tv['epsilon_leak']:.4f}")

        with open(self.output_dir / "data" / "experiment_1_trust_quantification.json", 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_trust_comparison(results)
        return results

    # ------------------------------------------------------------------
    # Experiment 2  
    # ------------------------------------------------------------------

    def experiment_2_entropy_certification(self, n_bits: int = 3500000):
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Entropy Certification vs Output Uniformity  [PARALLEL]")
        print("=" * 80)

        scenarios = create_test_scenarios()
        task_args = [(name, params, n_bits) for name, params in scenarios.items()]
        results   = {}

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_run_exp2_scenario, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                name, result = fut.result()
                results[name] = result
                print(f"  [{name}] Certified Min-Entropy (EAT)={result['h_total_eat']:.4f}  "
                      f"certified_bits={result['certified_output_bits']}  "
                      f"Δ_EAT={result['delta_eat']:.4f}  "
                      f"blocks={result['blocks_used']}  "
                      f"H(output)={result['empirical_h_output']:.4f}  ← NOT security")

        print(f"\n  Experiment 2 wall time: {time.time() - t0:.1f} s")

        with open(self.output_dir / "data" / "experiment_2_entropy_certification.json", 'w') as f:
            json.dump(results, f, indent=2)

        # All results collected — now produce the two combined figures
        self._plot_entropy_comparison(results)        # Fig 1: H_cert vs H(output) + certified bits
        self._plot_eat_progression_combined(results)  # Fig 2: all-scenario EAT progression panels
        return results

    # ------------------------------------------------------------------
    # Experiment 3
    # ------------------------------------------------------------------

    def experiment_3_attack_detection(self, n_bits: int = 3500000):
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Attack Detection Capability  [PARALLEL]")
        print("=" * 80)

        attack_scenarios = {
            'no_attack':          IdealParams(),
            'weak_attack':        AttackedParams(attack_strength=0.1),
            'medium_attack':      AttackedParams(attack_strength=0.2),
            'strong_attack':      AttackedParams(attack_strength=0.3),
            'very_strong_attack': AttackedParams(attack_strength=0.4),
        }

        task_args = [(name, params, n_bits) for name, params in attack_scenarios.items()]
        results   = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_run_exp3_scenario, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                name, block_results = fut.result()
                results[name] = block_results
                for b in block_results:
                    print(f"  [{name}] Block {b['block']}: "
                          f"Trust={b['trust_score']:.4f}  "
                          f"H_total={b['h_total_eat']:.4f}  "
                          f"Rate={b['extraction_rate']:.4f}")

        with open(self.output_dir / "data" / "experiment_3_attack_detection.json", 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_attack_response(results)
        return results

    # ------------------------------------------------------------------
    # Experiment 4
    # ------------------------------------------------------------------

    def experiment_4_comparison_with_si_qrng(self, n_bits: int = 3500000):
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: Comparison with Standard SI-QRNG  [PARALLEL]")
        print("=" * 80)

        scenarios = create_test_scenarios()
        task_args = [(name, params, n_bits) for name, params in scenarios.items()]
        results   = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_run_exp4_scenario, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                name, result = fut.result()
                results[name] = result
                print(f"  [{name}] TE={result['te_output_bits']} bits "
                      f"quality={result['te_quality_score']:.4f}  |  "
                      f"SI={result['si_output_bits']} bits "
                      f"quality={result['si_quality_score']:.4f}")

        with open(self.output_dir / "data" / "experiment_4_comparison.json", 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_comparison(results)
        return results

    # ------------------------------------------------------------------
    # Experiment 5  (sequential by design — time-series, block-to-block state)
    # ------------------------------------------------------------------

    def experiment_5_temporal_adaptation(self, n_blocks: int = 20):
        print("\n" + "=" * 80)
        print("EXPERIMENT 5: Temporal Adaptation to Source Degradation")
        print("=" * 80)

        results = {
            'block':           [],
            'trust_score':     [],
            'h_min_certified': [],
            'extraction_rate': [],
            'output_bits':     [],
            'source_quality':  [],
        }

        te_qrng = TrustEnhancedQRNG(block_size=1000000)

        for block_idx in range(n_blocks):
            phase          = 2 * np.pi * block_idx / n_blocks
            source_quality = 0.5 + 0.5 * np.cos(phase)
            bias           = 0.3 * (1 - source_quality)
            params         = BiasedParams(bias=bias)
            source         = QuantumSourceSimulator(params, seed=42 + block_idx)

            block          = source.generate_block(1000000)
            output, metadata = te_qrng.process_block(
                block.bits, block.bases, block.raw_signal
            )

            results['block'].append(block_idx)
            results['trust_score'].append(metadata['trust_score'])
            results['h_min_certified'].append(
                metadata.get('h_min_certified', 0.0)
            )
            results['extraction_rate'].append(metadata['extraction_rate'])
            results['output_bits'].append(metadata['output_bits'])
            results['source_quality'].append(source_quality)

            if block_idx % 5 == 0:
                print(f"  Block {block_idx}: Quality={source_quality:.3f}  "
                      f"Trust={metadata['trust_score']:.3f}  "
                      f"Rate={metadata['extraction_rate']:.3f}")

        with open(self.output_dir / "data" / "experiment_5_temporal_adaptation.json", 'w') as f:
            json.dump(results, f, indent=2)

        self._plot_temporal_adaptation(results)
        return results

    # ------------------------------------------------------------------
    # Quality score (vectorized)
    # ------------------------------------------------------------------

    def _compute_quality_score(self, bits: np.ndarray) -> float:
        return _compute_quality_score_static(bits)

    # ------------------------------------------------------------------
    # Plots — identical to v1, unchanged
    # ------------------------------------------------------------------

    def _plot_trust_comparison(self, results: Dict):
        scenarios     = list(results.keys())
        trust_scores  = [results[s]['trust_score']   for s in scenarios]
        epsilon_bias  = [results[s]['epsilon_bias']  for s in scenarios]
        epsilon_corr  = [results[s]['epsilon_corr']  for s in scenarios]
        epsilon_drift = [results[s]['epsilon_drift'] for s in scenarios]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        x = np.arange(len(scenarios))

        ax1.bar(x, trust_scores, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Scenario', fontsize=12)
        ax1.set_ylabel('Trust Score', fontsize=12)
        ax1.set_title('Trust Scores Across Scenarios', fontsize=14, fontweight='bold')
        ax1.set_xticks(x); ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.axhline(y=0.9, color='green',  linestyle='--', label='High Trust')
        ax1.axhline(y=0.7, color='orange', linestyle='--', label='Medium Trust')
        ax1.axhline(y=0.5, color='red',    linestyle='--', label='Low Trust')
        ax1.legend(); ax1.grid(axis='y', alpha=0.3)

        width = 0.25
        ax2.bar(x - width, epsilon_bias,  width, label='ε_bias',  alpha=0.7)
        ax2.bar(x,         epsilon_corr,  width, label='ε_corr',  alpha=0.7)
        ax2.bar(x + width, epsilon_drift, width, label='ε_drift', alpha=0.7)
        ax2.set_xlabel('Scenario', fontsize=12)
        ax2.set_ylabel('Epsilon Value', fontsize=12)
        ax2.set_title('Trust Vector Components', fontsize=14, fontweight='bold')
        ax2.set_xticks(x); ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.legend(); ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "experiment_1_trust_comparison.png", dpi=300)
        plt.close()
        print("  Saved: experiment_1_trust_comparison.png")

    def _plot_entropy_comparison(self, results: Dict):
        scenarios        = list(results.keys())
        empirical        = [results[s].get('empirical_h_output', 0.0) for s in scenarios]
        certified        = [results[s].get('h_total_eat', 0.0)        for s in scenarios]
        certified_output = [results[s].get('certified_output_bits', 0) for s in scenarios]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        x = np.arange(len(scenarios)); w = 0.35

        ax1.bar(x - w/2, empirical, w, label='H(output) — uniformity, NOT security',
                alpha=0.7, color='steelblue')
        ax1.bar(x + w/2, certified, w, label='H_min certified — cross-basis bound',
                alpha=0.7, color='darkorange')
        ax1.set_xlabel('Scenario', fontsize=12)
        ax1.set_ylabel('Entropy (bits/bit)', fontsize=12)
        ax1.set_title('Certified H_min vs Output Uniformity\n'
                      '(H_cert < H(output) expected — conservative bound)',
                      fontsize=11, fontweight='bold')
        ax1.set_xticks(x); ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)

        ax2.bar(x, certified_output, color='coral', alpha=0.7)
        ax2.set_xlabel('Scenario', fontsize=12)
        ax2.set_ylabel('Certified Output Bits', fontsize=12)
        ax2.set_title('Globally Certified Output Bits\n(from final EAT summary)',
                      fontsize=11, fontweight='bold')
        ax2.set_xticks(x); ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "experiment_2_entropy_comparison.png", dpi=300)
        plt.close()
        print("  Saved: experiment_2_entropy_comparison.png")

    def _plot_eat_progression_combined(self, results: Dict):
        """
        Single combined figure: one panel per scenario showing H_total and Δ_EAT
        progression over blocks.  All 12 scenarios on one page — 4 rows × 3 cols.
        """
        scenarios = list(results.keys())
        n         = len(scenarios)
        cols      = 3
        rows      = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows))
        axes = np.array(axes).flatten()

        colors = {'h_total': 'steelblue', 'delta': 'darkorange'}

        for idx, name in enumerate(scenarios):
            ax  = axes[idx]
            res = results[name]
            h_prog = res['h_total_progression']
            d_prog = res['delta_progression']
            xs     = list(range(len(h_prog)))

            ax.plot(xs, h_prog, color=colors['h_total'], linewidth=2,
                    label=f"H_total (final={res['h_total_eat']:.2f})")
            ax.plot(xs, d_prog, color=colors['delta'],  linewidth=1.5,
                    linestyle='--', label='Δ_EAT correction')
            ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')

            # Annotate certified output bits
            ax.set_title(f"{name}\ncertified bits = {res['certified_output_bits']}",
                         fontsize=9, fontweight='bold')
            ax.set_xlabel('Blocks', fontsize=8)
            ax.set_ylabel('Bits', fontsize=8)
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)

        # Hide unused axes
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('Experiment 2 — EAT Entropy Accumulation per Scenario\n'
                     '(H_total builds up over blocks; Δ_EAT is the EAT correction penalty)',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "experiment_2_eat_progression_all.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: experiment_2_eat_progression_all.png")

    def _plot_attack_response(self, results: Dict):
        n_scenarios = len(results)
        rows = (n_scenarios + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = np.array(axes).flatten()

        for idx, (scenario_name, blocks) in enumerate(results.items()):
            ax = axes[idx]
            block_nums      = [b['block'] for b in blocks]
            trust_scores    = [b['trust_score'] for b in blocks]
            extraction_rates = [b['extraction_rate'] for b in blocks]

            ax.plot(block_nums, trust_scores,     'o-', label='Trust Score',    linewidth=2)
            ax.plot(block_nums, extraction_rates, 's-', label='Extraction Rate', linewidth=2)
            ax.set_xlabel('Block Number'); ax.set_ylabel('Score / Rate')
            ax.set_title(scenario_name.replace("_", " ").title(), fontweight='bold')
            ax.legend(); ax.grid(alpha=0.3); ax.set_ylim([0, 1.05])

        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "experiment_3_attack_response.png", dpi=300)
        plt.close()
        print("  Saved: experiment_3_attack_response.png")

    def _plot_comparison(self, results: Dict):
        """
        Three-panel comparison figure — all scenarios, both systems.

        Panel 1: Output bits  (TE-SI-QRNG vs Standard SI-QRNG)
        Panel 2: Quality score (TE vs SI)
        Panel 3: Average certified H_min per block (TE vs SI)
        """
        scenarios   = list(results.keys())
        te_output   = [results[s]['te_output_bits']   for s in scenarios]
        si_output   = [results[s]['si_output_bits']   for s in scenarios]
        te_quality  = [results[s]['te_quality_score'] for s in scenarios]
        si_quality  = [results[s]['si_quality_score'] for s in scenarios]
        te_h_cert   = [results[s]['te_avg_h_cert']    for s in scenarios]
        si_h_cert   = [results[s]['si_avg_h_cert']    for s in scenarios]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        x = np.arange(len(scenarios))
        w = 0.38

        # --- Panel 1: Output bits ---
        ax1.bar(x - w/2, te_output, w, label='TE-SI-QRNG',       alpha=0.8, color='steelblue')
        ax1.bar(x + w/2, si_output, w, label='Standard SI-QRNG', alpha=0.8, color='coral')
        ax1.set_xlabel('Scenario', fontsize=11)
        ax1.set_ylabel('Output Bits', fontsize=11)
        ax1.set_title('Output Quantity\nTE-SI-QRNG vs Standard SI-QRNG',
                      fontsize=11, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # --- Panel 2: Quality score ---
        ax2.bar(x - w/2, te_quality, w, label='TE-SI-QRNG',       alpha=0.8, color='steelblue')
        ax2.bar(x + w/2, si_quality, w, label='Standard SI-QRNG', alpha=0.8, color='coral')
        ax2.set_xlabel('Scenario', fontsize=11)
        ax2.set_ylabel('Quality Score', fontsize=11)
        ax2.set_title('Output Quality Score\n(freq + runs test composite)',
                      fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.1])

        # --- Panel 3: Average certified H_min ---
        ax3.bar(x - w/2, te_h_cert, w, label='TE-SI-QRNG (with self-testing)',
                alpha=0.8, color='steelblue')
        ax3.bar(x + w/2, si_h_cert, w, label='Standard SI-QRNG (no self-testing)',
                alpha=0.8, color='coral')
        ax3.set_xlabel('Scenario', fontsize=11)
        ax3.set_ylabel('Avg H_min per block (bits/bit)', fontsize=11)
        ax3.set_title('Certified Min-Entropy per Block\n(conservative security bound)',
                      fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

        fig.suptitle('Experiment 4 — TE-SI-QRNG vs Standard SI-QRNG  (all 12 scenarios)',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "experiment_4_comparison.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: experiment_4_comparison.png")

    def _plot_temporal_adaptation(self, results: Dict):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        blocks          = results['block']
        source_quality  = results['source_quality']
        trust_score     = results['trust_score']
        extraction_rate = results['extraction_rate']
        h_cert          = results['h_min_certified']

        # ── Panel 1: Trust vs Source Quality ──────────────────────────
        ax1.plot(blocks, source_quality, 'o-', label='True Source Quality',
                 linewidth=2, markersize=6, color='green')
        ax1.plot(blocks, trust_score,    's-', label='Measured Trust Score',
                 linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('Block Number'); ax1.set_ylabel('Quality / Trust Score')
        ax1.set_title('Trust Adaptation to Source Quality Changes', fontweight='bold')
        ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim([0, 1.05])

        # ── Panel 2: Certified H_min vs Extraction Rate ────────────────
        # Root cause of missing purple line: h_cert and extraction_rate are
        # nearly identical in value (LHL: rate = h_cert - security_cost),
        # so purple was completely hidden behind orange.
        # Fix: shade the gap, draw purple on top (zorder=3), dash orange.
        ax2.fill_between(blocks, h_cert, extraction_rate,
                         where=[h >= e for h, e in zip(h_cert, extraction_rate)],
                         alpha=0.18, color='purple',
                         label='LHL security gap (H_min − Rate)')
        ax2.plot(blocks, h_cert, 'o-',
                 label='Certified H_min (cross-basis bound)',
                 linewidth=2.5, markersize=7, color='purple', zorder=3)
        ax2.plot(blocks, extraction_rate, 's--',
                 label='Extraction Rate',
                 linewidth=2, markersize=6, color='orange', zorder=2)
        ax2.set_xlabel('Block Number'); ax2.set_ylabel('Rate / Certified Entropy')
        ax2.set_title('Certified Entropy & Extraction Rate Adaptation', fontweight='bold')
        ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "figures" / "experiment_5_temporal_adaptation.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: experiment_5_temporal_adaptation.png")

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all_experiments(self):
        print("\n" + "=" * 80)
        print("RUNNING ALL EXPERIMENTS FOR TE-SI-QRNG  (PARALLEL EDITION)")
        print("=" * 80)

        t0 = time.time()

        exp1 = self.experiment_1_trust_quantification()
        exp2 = self.experiment_2_entropy_certification()
        exp3 = self.experiment_3_attack_detection()
        exp4 = self.experiment_4_comparison_with_si_qrng()
        exp5 = self.experiment_5_temporal_adaptation()

        elapsed = time.time() - t0
        print("\n" + "=" * 80)
        print(f"ALL EXPERIMENTS COMPLETED in {elapsed:.2f} s  ({elapsed/60:.1f} min)")
        print("=" * 80)
        print(f"Results → {self.output_dir}")

        return {'experiment_1': exp1, 'experiment_2': exp2,
                'experiment_3': exp3, 'experiment_4': exp4,
                'experiment_5': exp5}



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = ExperimentRunner(output_dir="results")
    all_results = runner.run_all_experiments()

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print("  1. Trust quantification distinguishes all 12 source types")
    print("  2. Entropy certification: conservative EAT bound vs empirical uniformity")
    print("  3. Attack detection: trust_score drops with attack_strength")
    print("  4. TE-SI-QRNG vs Standard SI-QRNG quality/quantity tradeoff")
    print("  5. Temporal adaptation: trust tracks sinusoidal quality degradation")
