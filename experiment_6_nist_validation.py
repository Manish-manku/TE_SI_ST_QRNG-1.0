"""
Experiment 6 — NIST SP 800-22 Randomness Validation
=====================================================

Validates TE-SI-QRNG output against the full NIST SP 800-22 Rev 1a battery
(15 statistical tests).  Produces four publication-quality figures:

  Fig 6-A  P-value heatmap  — 15 tests × 12 scenarios (post-extraction)
  Fig 6-B  Pre vs Post extraction — side-by-side heatmaps, all 12 scenarios
  Fig 6-C  Pass/fail summary table — scenario rows × test columns + pass-rate
  Fig 6-D  Attack spotlight — NIST pass rate + trust score vs attack strength

Usage
-----
    python experiment_6_nist_validation.py

Or from experiment_v2_1.py:
    from experiment_6_nist_validation import NISTExperimentRunner
    nist_runner = NISTExperimentRunner(output_dir="results")
    nist_runner.run_all()
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

from D_v2 import TrustEnhancedQRNG
from New_simulator import (
    QuantumSourceSimulator,
    create_test_scenarios,
    IdealParams,
    AttackedParams,
)
from nist_runner import NISTTestRunner, NISTResult, TEST_NAMES, N_TESTS


# ---------------------------------------------------------------------------
# Parallel worker functions (top-level for pickling)
# ---------------------------------------------------------------------------

def _worker_post_extraction(args) -> Tuple[str, Dict]:
    """
    Worker: generate extracted bits for one scenario and run all NIST tests.
    Returns (scenario_name, result_dict).
    """
    scenario_name, params, n_bits = args

    source  = QuantumSourceSimulator(params, seed=42)
    te_qrng = TrustEnhancedQRNG(block_size=1_000_000)

    output_bits, metadata_list = te_qrng.generate_certified_random_bits(
        n_bits=n_bits, source_simulator=source
    )

    runner = NISTTestRunner(significance=0.01)

    if len(output_bits) >= 1000:
        nist_result = runner.run_all(output_bits[:n_bits])
    else:
        # Not enough bits — return zeroed result
        nist_result = NISTResult(
            p_values=[None] * N_TESTS,
            passed=[None] * N_TESTS,
            n_bits=len(output_bits),
        )

    final_meta = metadata_list[-1] if metadata_list else {}

    return scenario_name, {
        "p_values":    nist_result.p_values,
        "passed":      nist_result.passed,
        "pass_rate":   nist_result.pass_rate(),
        "n_bits":      len(output_bits),
        "h_total_eat": final_meta.get("h_total_eat", 0.0),
        "backend":     nist_result.backend,
    }


def _worker_pre_extraction(args) -> Tuple[str, Dict]:
    """
    Worker: generate RAW bits for one scenario (no extraction) and run NIST.
    """
    scenario_name, params, n_bits = args

    source = QuantumSourceSimulator(params, seed=42)
    block  = source.generate_block(n_bits)
    raw_bits = block.bits[:n_bits]

    runner      = NISTTestRunner(significance=0.01)
    nist_result = runner.run_all(raw_bits)

    return scenario_name, {
        "p_values":  nist_result.p_values,
        "passed":    nist_result.passed,
        "pass_rate": nist_result.pass_rate(),
        "n_bits":    len(raw_bits),
        "backend":   nist_result.backend,
    }


def _worker_attack_sweep(args) -> Tuple[float, Dict]:
    """
    Worker: one attack-strength value — run NIST on both raw and extracted bits.
    """
    attack_strength, n_bits = args

    params  = AttackedParams(attack_strength=attack_strength)
    source  = QuantumSourceSimulator(params, seed=42)
    te_qrng = TrustEnhancedQRNG(block_size=1_000_000)

    # Raw bits
    raw_block = source.generate_block(n_bits)
    raw_bits  = raw_block.bits[:n_bits]

    # Extracted bits
    source.reset()
    output_bits, metadata_list = te_qrng.generate_certified_random_bits(
        n_bits=n_bits, source_simulator=source
    )

    runner = NISTTestRunner(significance=0.01)

    raw_result = runner.run_all(raw_bits)
    ext_result = (runner.run_all(output_bits[:n_bits])
                  if len(output_bits) >= 1000
                  else NISTResult(p_values=[None]*N_TESTS, passed=[None]*N_TESTS, n_bits=0))

    # Trust score from last block metadata
    block_metas = [m for m in metadata_list if "trust_score" in m]
    trust_score = float(np.mean([m["trust_score"] for m in block_metas])) if block_metas else 0.0

    return attack_strength, {
        "raw_pass_rate":  raw_result.pass_rate(),
        "ext_pass_rate":  ext_result.pass_rate(),
        "trust_score":    trust_score,
        "raw_p_values":   raw_result.p_values,
        "ext_p_values":   ext_result.p_values,
    }


# ---------------------------------------------------------------------------
# NISTExperimentRunner
# ---------------------------------------------------------------------------

class NISTExperimentRunner:
    """
    Orchestrates all NIST SP 800-22 experiments and produces figures.

    Experiments
    -----------
    6-A  P-value heatmap: 15 tests × 12 scenarios (post-extraction)
    6-B  Pre vs Post extraction: side-by-side pass/fail heatmaps
    6-C  Pass/fail summary table with pass-rate column
    6-D  Attack spotlight: NIST pass rate + trust score vs attack strength
    """

    # ------------------------------------------------------------------ init
    def __init__(self,
                 output_dir:  str = "results",
                 n_bits:      int = 1_000_000,
                 max_workers: Optional[int] = None):
        self.output_dir  = Path(output_dir)
        self.n_bits      = n_bits
        self.max_workers = max_workers or os.cpu_count()

        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        print(f"[NISTExperimentRunner]  n_bits={n_bits:,}  "
              f"workers={self.max_workers}  output={self.output_dir}")

    # ------------------------------------------------------------------ run all
    def run_all(self) -> Dict:
        print("\n" + "=" * 80)
        print("EXPERIMENT 6: NIST SP 800-22 RANDOMNESS VALIDATION")
        print("=" * 80)
        t0 = time.time()

        post_results = self._collect_post_extraction()
        pre_results  = self._collect_pre_extraction()
        attack_sweep = self._collect_attack_sweep()

        self._plot_6A_pvalue_heatmap(post_results)
        self._plot_6B_pre_post_heatmap(pre_results, post_results)
        self._plot_6C_passfail_table(post_results)
        self._plot_6D_attack_spotlight(attack_sweep)

        elapsed = time.time() - t0
        print(f"\nExperiment 6 completed in {elapsed:.1f} s")

        all_results = {
            "post_extraction": post_results,
            "pre_extraction":  pre_results,
            "attack_sweep":    attack_sweep,
        }
        with open(self.output_dir / "data" / "experiment_6_nist.json", "w") as f:
            json.dump(all_results, f, indent=2)

        return all_results

    # ------------------------------------------------------------------ data collection
    def _collect_post_extraction(self) -> Dict:
        print("\n[6-A / 6-C] Running NIST tests on extracted output — all 12 scenarios")
        scenarios = create_test_scenarios()
        task_args = [(n, p, self.n_bits) for n, p in scenarios.items()]
        return self._run_parallel(_worker_post_extraction, task_args, "post")

    def _collect_pre_extraction(self) -> Dict:
        print("\n[6-B] Running NIST tests on RAW bits — all 12 scenarios")
        scenarios = create_test_scenarios()
        task_args = [(n, p, self.n_bits) for n, p in scenarios.items()]
        return self._run_parallel(_worker_pre_extraction, task_args, "pre")

    def _collect_attack_sweep(self) -> Dict:
        print("\n[6-D] Attack sweep: NIST pass rate vs attack_strength")
        strengths = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        task_args = [(s, self.n_bits) for s in strengths]
        results   = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_worker_attack_sweep, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                strength, result = fut.result()
                results[strength] = result
                print(f"  attack={strength:.2f}  raw_pass={result['raw_pass_rate']:.1%}  "
                      f"ext_pass={result['ext_pass_rate']:.1%}  "
                      f"trust={result['trust_score']:.3f}")
        return {str(k): v for k, v in sorted(results.items())}

    def _run_parallel(self, worker_fn, task_args, label: str) -> Dict:
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(worker_fn, a): a[0] for a in task_args}
            for fut in as_completed(futures):
                name, result = fut.result()
                results[name] = result
                print(f"  [{label}] {name:25s}  pass_rate={result['pass_rate']:.1%}  "
                      f"n_bits={result['n_bits']:,}")
        return results

    # ------------------------------------------------------------------ Fig 6-A
    def _plot_6A_pvalue_heatmap(self, post_results: Dict):
        """
        P-value heatmap — rows = 15 NIST tests, columns = 12 scenarios.
        Colour encodes p-value (0→1).  Red zone below α=0.01 = FAIL.
        """
        scenarios = list(post_results.keys())
        n_sc      = len(scenarios)

        # Build p-value matrix  (N_TESTS × n_scenarios)
        pmat = np.full((N_TESTS, n_sc), np.nan)
        for j, sc in enumerate(scenarios):
            for i, pv in enumerate(post_results[sc]["p_values"]):
                if pv is not None:
                    pmat[i, j] = pv

        fig, ax = plt.subplots(figsize=(max(14, n_sc * 1.1), 10))

        # Custom colormap: red→yellow→green
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "nist", ["#d32f2f", "#ffeb3b", "#388e3c"], N=256
        )
        cmap.set_bad(color="#bdbdbd")    # grey for skipped

        im = ax.imshow(pmat, cmap=cmap, vmin=0.0, vmax=1.0,
                       aspect="auto", interpolation="nearest")

        # Significance threshold line overlay
        # Draw a horizontal white threshold band at p = 0.01
        ax.axhline(-0.5, color="none")   # dummy to set limits properly

        # Annotate cells
        for i in range(N_TESTS):
            for j in range(n_sc):
                pv = pmat[i, j]
                if np.isnan(pv):
                    ax.text(j, i, "SKIP", ha="center", va="center",
                            fontsize=6.5, color="#555555", fontweight="bold")
                else:
                    label = f"{pv:.3f}"
                    color = "white" if pv < 0.05 else "black"
                    fw    = "bold" if pv < 0.01 else "normal"
                    ax.text(j, i, label, ha="center", va="center",
                            fontsize=6.5, color=color, fontweight=fw)

        # Axes
        ax.set_xticks(range(n_sc))
        ax.set_xticklabels(scenarios, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(N_TESTS))
        ax.set_yticklabels(TEST_NAMES, fontsize=9)
        ax.set_xlabel("Source Scenario", fontsize=11, labelpad=8)
        ax.set_title(
            "Figure 6-A  —  NIST SP 800-22 P-Value Heatmap (Post-Extraction)\n"
            "Colour: p-value  │  Red = FAIL (p < 0.01)  │  Grey = SKIP (insufficient cycles)",
            fontsize=11, fontweight="bold", pad=10
        )

        cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
        cbar.set_label("p-value", fontsize=10)
        cbar.ax.axhline(y=0.01, color="red", linewidth=2, label="α=0.01")
        cbar.ax.text(2.2, 0.01, "α=0.01", color="red", fontsize=8, va="center")

        # Pass-rate row below the heatmap (text annotation)
        for j, sc in enumerate(scenarios):
            pr = post_results[sc]["pass_rate"]
            color = "#1b5e20" if pr >= 0.80 else ("#e65100" if pr >= 0.60 else "#b71c1c")
            ax.text(j, N_TESTS - 0.05, f"{pr:.0%}",
                    ha="center", va="bottom", fontsize=8,
                    color=color, fontweight="bold",
                    transform=ax.get_xaxis_transform())

        plt.tight_layout()
        fpath = self.output_dir / "figures" / "experiment_6A_pvalue_heatmap.png"
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath.name}")

    # ------------------------------------------------------------------ Fig 6-B
    def _plot_6B_pre_post_heatmap(self, pre_results: Dict, post_results: Dict):
        """
        Side-by-side pass/fail heatmaps: left = raw bits, right = extracted bits.
        Rows = scenarios, columns = 15 tests.
        """
        # Use intersection of scenarios present in both
        scenarios = [s for s in pre_results if s in post_results]
        n_sc      = len(scenarios)

        def build_pass_matrix(results):
            mat = np.full((n_sc, N_TESTS), np.nan)
            for i, sc in enumerate(scenarios):
                for j, pv in enumerate(results[sc]["passed"]):
                    if pv is not None:
                        mat[i, j] = float(pv)
            return mat

        pre_mat  = build_pass_matrix(pre_results)
        post_mat = build_pass_matrix(post_results)

        # Colour: 1=green (pass), 0=red (fail), nan=grey (skip)
        cmap = mcolors.ListedColormap(["#d32f2f", "#388e3c"])
        cmap.set_bad("#bdbdbd")

        fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(24, max(8, n_sc * 0.65 + 2)))

        for ax, mat, title in [
            (ax_pre,  pre_mat,  "PRE-Extraction  (raw quantum bits)"),
            (ax_post, post_mat, "POST-Extraction  (Toeplitz hashed output)"),
        ]:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")

            for i in range(n_sc):
                for j in range(N_TESTS):
                    v = mat[i, j]
                    if np.isnan(v):
                        ax.text(j, i, "—", ha="center", va="center",
                                fontsize=7, color="#777777")
                    else:
                        txt = "✓" if v == 1 else "✗"
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=9, color="white", fontweight="bold")

            ax.set_xticks(range(N_TESTS))
            ax.set_xticklabels(
                [n.split("(")[0].strip() for n in TEST_NAMES],
                rotation=55, ha="right", fontsize=8
            )
            ax.set_yticks(range(n_sc))
            ax.set_yticklabels(scenarios, fontsize=8)
            ax.set_title(f"Figure 6-B  —  {title}", fontsize=10, fontweight="bold")

            # Pass-rate column on the right
            for i, sc in enumerate(scenarios):
                pr = results[sc]["pass_rate"] if (results := (
                    pre_results if ax is ax_pre else post_results
                )) else 0.0
                color = "#1b5e20" if pr >= 0.80 else "#b71c1c"
                ax.text(N_TESTS + 0.1, i, f"{pr:.0%}",
                        ha="left", va="center", fontsize=8,
                        color=color, fontweight="bold",
                        transform=ax.transData)

        legend_patches = [
            mpatches.Patch(color="#388e3c", label="PASS  (p ≥ 0.01)"),
            mpatches.Patch(color="#d32f2f", label="FAIL  (p < 0.01)"),
            mpatches.Patch(color="#bdbdbd", label="SKIP  (insufficient cycles)"),
        ]
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(
            "Figure 6-B  —  NIST SP 800-22 Pass/Fail: Pre-Extraction vs Post-Extraction\n"
            "The Toeplitz extractor whitens imperfect sources — attacked/biased inputs\n"
            "may still pass NIST after extraction, confirming extractor effectiveness.",
            fontsize=11, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        fpath = self.output_dir / "figures" / "experiment_6B_pre_post_heatmap.png"
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath.name}")

    # ------------------------------------------------------------------ Fig 6-C
    def _plot_6C_passfail_table(self, post_results: Dict):
        """
        Styled pass/fail table: scenario rows × test columns + pass-rate.
        Rendered as a matplotlib table figure — no pandas needed.
        """
        scenarios = list(post_results.keys())
        n_sc      = len(scenarios)

        # Short column headers to fit
        short_names = [
            "Freq", "Blk-Freq", "Runs", "LngRun", "MatRank",
            "DFT", "Non-OvTpl", "OvTpl", "Universal", "LinCmpx",
            "Serial", "ApxEnt", "CumSums", "RndExc", "RndExcV",
        ]

        # Cell values and colours
        cell_text   = []
        cell_colors = []

        for sc in scenarios:
            row_text   = []
            row_colors = []
            for pv, ok in zip(post_results[sc]["p_values"],
                              post_results[sc]["passed"]):
                if pv is None:
                    row_text.append("SKIP")
                    row_colors.append("#e0e0e0")
                elif ok:
                    row_text.append(f"{pv:.3f}")
                    row_colors.append("#c8e6c9")   # light green
                else:
                    row_text.append(f"{pv:.3f}")
                    row_colors.append("#ffcdd2")   # light red
            # Pass-rate column
            pr = post_results[sc]["pass_rate"]
            row_text.append(f"{pr:.0%}")
            row_colors.append("#fff9c4" if pr >= 0.80 else "#ffccbc")
            cell_text.append(row_text)
            cell_colors.append(row_colors)

        col_labels = short_names + ["Rate"]

        fig_width  = max(20, len(col_labels) * 1.25)
        fig_height = max(4,  n_sc * 0.55 + 2.5)
        fig, ax    = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=scenarios,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.6)

        # Bold header row
        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7)

        # Bold row labels
        for i in range(n_sc):
            cell = tbl[i + 1, -1]
            cell.set_text_props(fontweight="bold")

        ax.set_title(
            "Figure 6-C  —  NIST SP 800-22 Pass/Fail Table  (Post-Extraction, α = 0.01)\n"
            "Green = PASS  │  Red = FAIL  │  Grey = SKIP  │  Rate = fraction of tests passed",
            fontsize=10, fontweight="bold", pad=14
        )

        plt.tight_layout()
        fpath = self.output_dir / "figures" / "experiment_6C_passfail_table.png"
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath.name}")

    # ------------------------------------------------------------------ Fig 6-D
    def _plot_6D_attack_spotlight(self, attack_sweep: Dict):
        """
        Attack spotlight: three lines vs attack_strength.
          • NIST pass rate on RAW bits     (blue solid)
          • NIST pass rate on EXTRACTED bits (green dashed)
          • Trust score                    (orange dotted)

        Key insight visualised:
          Raw bits fail NIST as attack grows.
          Extracted bits may still pass — extractor whitens the output.
          Trust score drops — correctly diagnosing the attack.
        """
        strengths   = sorted([float(k) for k in attack_sweep.keys()])
        raw_rates   = [attack_sweep[str(s)]["raw_pass_rate"]  for s in strengths]
        ext_rates   = [attack_sweep[str(s)]["ext_pass_rate"]  for s in strengths]
        trust_scores= [attack_sweep[str(s)]["trust_score"]    for s in strengths]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        lw = 2.5
        l1, = ax1.plot(strengths, raw_rates,    "o-",  color="#1565c0",
                       linewidth=lw, markersize=8, label="NIST pass rate — RAW bits")
        l2, = ax1.plot(strengths, ext_rates,    "s--", color="#2e7d32",
                       linewidth=lw, markersize=8, label="NIST pass rate — Extracted bits")
        l3, = ax2.plot(strengths, trust_scores, "^:",  color="#e65100",
                       linewidth=lw, markersize=8, label="Trust Score (diagnostic)")

        # Significance shading
        ax1.axhspan(0.0, 0.80, alpha=0.05, color="red")
        ax1.axhline(0.80, color="#b71c1c", linewidth=1.0, linestyle=":", alpha=0.6)
        ax1.text(0.01, 0.81, "80 % pass threshold", color="#b71c1c",
                 fontsize=8, alpha=0.8)

        ax2.axhline(DiagnosticHaltThreshold := 0.5, color="#e65100",
                    linewidth=1.0, linestyle=":", alpha=0.5)
        ax2.text(strengths[-1] * 0.55, 0.52, "warn threshold (0.5)",
                 color="#e65100", fontsize=8, alpha=0.7)

        ax1.set_xlabel("Attack Strength", fontsize=12)
        ax1.set_ylabel("NIST Pass Rate (fraction)", fontsize=11, color="#1565c0")
        ax2.set_ylabel("Trust Score", fontsize=11, color="#e65100")
        ax1.set_ylim(-0.05, 1.10)
        ax2.set_ylim(-0.05, 1.10)
        ax1.tick_params(axis="y", labelcolor="#1565c0")
        ax2.tick_params(axis="y", labelcolor="#e65100")

        lines  = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="lower left", fontsize=9)
        ax1.grid(alpha=0.3)

        ax1.set_title(
            "Figure 6-D  —  NIST Pass Rate & Trust Score vs Attack Strength\n"
            "Raw bits fail NIST as attack grows  │  Extractor whitens output (green stays high)\n"
            "Trust score correctly diagnoses attack even when NIST on extracted bits still passes",
            fontsize=10, fontweight="bold"
        )

        plt.tight_layout()
        fpath = self.output_dir / "figures" / "experiment_6D_attack_spotlight.png"
        plt.savefig(fpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath.name}")


# ---------------------------------------------------------------------------
# NIST Summary for Experiment 2 (imported by experiment_v2_1.py)
# ---------------------------------------------------------------------------

def nist_summary_for_experiment_2(post_results: Dict,
                                   output_dir: Path) -> None:
    """
    Produce a compact NIST summary figure suitable for inclusion in
    Experiment 2's entropy certification section.

    Single-row heatmap: 12 scenarios × pass-rate + mini p-value bar.
    Saved as experiment_2_nist_summary.png.
    """
    scenarios  = list(post_results.keys())
    pass_rates = [post_results[s]["pass_rate"] for s in scenarios]

    fig, ax = plt.subplots(figsize=(14, 3))

    colors = ["#388e3c" if pr >= 0.80 else
              "#f57c00" if pr >= 0.60 else "#d32f2f"
              for pr in pass_rates]

    bars = ax.bar(scenarios, pass_rates, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0.80, color="grey", linewidth=1, linestyle="--", alpha=0.7,
               label="80 % threshold")
    ax.axhline(1.00, color="#1b5e20", linewidth=0.5, linestyle=":")

    for bar, pr in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{pr:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("NIST Pass Rate", fontsize=10)
    ax.set_title(
        "Experiment 2 Supplement  —  NIST SP 800-22 Pass Rate per Scenario (Post-Extraction)\n"
        "(Full details in Experiment 6)",
        fontsize=10, fontweight="bold"
    )
    ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fpath = output_dir / "figures" / "experiment_2_nist_summary.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner = NISTExperimentRunner(
        output_dir="results",
        n_bits=1_000_000,
    )
    results = runner.run_all()

    print("\n" + "=" * 80)
    print("EXPERIMENT 6 SUMMARY")
    print("=" * 80)

    post = results["post_extraction"]
    scenarios_sorted = sorted(post.keys(), key=lambda s: post[s]["pass_rate"], reverse=True)

    print("\nPost-extraction NIST pass rates (sorted):")
    for sc in scenarios_sorted:
        pr = post[sc]["pass_rate"]
        marker = "✓" if pr >= 0.80 else "✗"
        print(f"  {marker}  {sc:25s}  {pr:.1%}")

    overall = np.mean([post[s]["pass_rate"] for s in post])
    print(f"\nOverall mean pass rate: {overall:.1%}")
    print("\nKey finding:")
    print("  • Ideal / phase-noise sources → high NIST pass rates after extraction")
    print("  • Strongly attacked sources  → raw bits fail NIST, extracted bits")
    print("    may still pass (extractor whitens), but trust score drops sharply")
    print("  • Trust score is the correct security indicator — NIST is supplementary")
