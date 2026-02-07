#!/usr/bin/env python3
"""
VICEROY 2026 — Publication Figure Generator
============================================

Reads JSON results from results/ and produces publication-quality figures
for the V6 algorithmic benchmark and hardware emulation Digital Twin.

Outputs:
    figures/v6_accuracy_vs_snr.png         — HDC vs MLP accuracy across SNR
    figures/hardware_robustness.png         — Noise sweep: graceful degradation
    figures/energy_comparison.png           — Energy per inference (CPU vs IMC)
    figures/viceroy_summary_dashboard.png   — Combined 2×2 dashboard

Usage:
    python src/generate_figures.py
"""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

V6_RESULTS = os.path.join(RESULTS_DIR, "v6_final_test.json")
HW_RESULTS = os.path.join(RESULTS_DIR, "hardware_emulation_results.json")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLORS = {
    "hdc":        "#1b9e77",   # Teal-green
    "mlp":        "#d95f02",   # Burnt orange
    "stuck_at":   "#7570b3",   # Muted purple
    "analog":     "#e7298a",   # Magenta-pink
    "combined":   "#1b9e77",   # Teal (same as HDC)
    "cpu":        "#636363",   # Gray
    "imc":        "#1b9e77",   # Teal
    "chance":     "#999999",   # Light gray
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def load_json(path):
    """Load a JSON file or exit with a clear message."""
    if not os.path.exists(path):
        print(f"  [SKIP] {os.path.basename(path)} not found.")
        return None
    with open(path, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Accuracy vs. SNR  (V6 Algorithmic Benchmark)
# ═══════════════════════════════════════════════════════════════════════════
def plot_accuracy_vs_snr(v6, ax=None):
    """HDC-RFF vs Steel Man MLP accuracy across all SNR bins."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4.5))

    hdc_snr = v6["hdc_rff"]["accuracy_by_snr"]
    mlp_snr = v6["steel_mlp"]["accuracy_by_snr"]

    snr_vals = sorted(hdc_snr.keys(), key=lambda x: int(x))
    snr_int = [int(s) for s in snr_vals]
    hdc_acc = [hdc_snr[s] for s in snr_vals]
    mlp_acc = [mlp_snr[s] for s in snr_vals]

    ax.plot(snr_int, hdc_acc, "o-", color=COLORS["hdc"], linewidth=2,
            markersize=5, label=v6["hdc_rff"]["model"], zorder=3)
    ax.plot(snr_int, mlp_acc, "s-", color=COLORS["mlp"], linewidth=2,
            markersize=5, label=v6["steel_mlp"]["model"], zorder=3)

    # Chance line
    n_classes = len(v6["configuration"]["modulations"])
    chance = 100.0 / n_classes
    ax.axhline(chance, color=COLORS["chance"], linestyle="--", linewidth=1,
               label=f"Chance ({chance:.0f}%)", zorder=1)

    # High-SNR threshold
    ax.axvline(10, color="#cccccc", linestyle=":", linewidth=1, zorder=1)
    ax.text(10.3, 28, "High-SNR\nthreshold", fontsize=8, color="#999999",
            va="center")

    # Annotations for high-SNR accuracy
    hdc_high = v6["hdc_rff"]["accuracy_high_snr_percent"]
    mlp_high = v6["steel_mlp"]["accuracy_high_snr_percent"]
    ax.annotate(f"{mlp_high:.1f}%", xy=(12, mlp_acc[snr_vals.index("12")]),
                xytext=(6, 88), fontsize=9, color=COLORS["mlp"],
                arrowprops=dict(arrowstyle="->", color=COLORS["mlp"], lw=1.2),
                fontweight="bold")
    ax.annotate(f"{hdc_high:.1f}%", xy=(12, hdc_acc[snr_vals.index("12")]),
                xytext=(0, 40), fontsize=9, color=COLORS["hdc"],
                arrowprops=dict(arrowstyle="->", color=COLORS["hdc"], lw=1.2),
                fontweight="bold")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Algorithmic Benchmark — HDC-RFF vs Steel Man MLP\n"
                 "(RadioML 2016.10A, 5 Tactical Modulations)")
    ax.set_ylim(0, 100)
    ax.set_xlim(snr_int[0] - 1, snr_int[-1] + 1)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    if standalone:
        out = os.path.join(FIGURES_DIR, "v6_accuracy_vs_snr.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  [SAVED] {out}")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Hardware Robustness  (Noise Sweep)
# ═══════════════════════════════════════════════════════════════════════════
def plot_hardware_robustness(hw, ax=None):
    """HDC & MLP noise sweep — stuck-at, analog, combined — apples-to-apples."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5.5))

    sweep = hw["noise_sweep_detailed"]
    noise_pct = [x * 100 for x in sweep["noise_levels_fraction"]]

    # --- HDC curves ---
    stuck_mean = [x * 100 for x in sweep["stuck_at"]["accuracy_mean"]]
    stuck_std  = [x * 100 for x in sweep["stuck_at"]["accuracy_std"]]
    analog_mean = [x * 100 for x in sweep["analog_noise"]["accuracy_mean"]]
    analog_std  = [x * 100 for x in sweep["analog_noise"]["accuracy_std"]]
    combined_mean = [x * 100 for x in sweep["combined"]["accuracy_mean"]]
    combined_std  = [x * 100 for x in sweep["combined"]["accuracy_std"]]

    # HDC error bands + lines
    ax.fill_between(noise_pct,
                    np.array(stuck_mean) - np.array(stuck_std),
                    np.array(stuck_mean) + np.array(stuck_std),
                    alpha=0.12, color=COLORS["stuck_at"])
    ax.plot(noise_pct, stuck_mean, "^-", color=COLORS["stuck_at"],
            linewidth=1.5, markersize=5, label="HDC — Stuck-at")

    ax.fill_between(noise_pct,
                    np.array(analog_mean) - np.array(analog_std),
                    np.array(analog_mean) + np.array(analog_std),
                    alpha=0.12, color=COLORS["analog"])
    ax.plot(noise_pct, analog_mean, "d-", color=COLORS["analog"],
            linewidth=1.5, markersize=5, label="HDC — Analog noise")

    ax.fill_between(noise_pct,
                    np.array(combined_mean) - np.array(combined_std),
                    np.array(combined_mean) + np.array(combined_std),
                    alpha=0.15, color=COLORS["combined"])
    ax.plot(noise_pct, combined_mean, "o-", color=COLORS["combined"],
            linewidth=2.2, markersize=6, label="HDC — Combined", zorder=4)

    # --- MLP curves (empirical if available, else estimated) ---
    mlp_sweep = hw.get("mlp_sweep_detailed")
    if mlp_sweep:
        # Full empirical MLP breakdown — same simulator, same methodology
        mlp_stuck_mean = [x * 100 for x in mlp_sweep["stuck_at"]["accuracy_mean"]]
        mlp_stuck_std  = [x * 100 for x in mlp_sweep["stuck_at"]["accuracy_std"]]
        mlp_analog_mean = [x * 100 for x in mlp_sweep["analog_noise"]["accuracy_mean"]]
        mlp_analog_std  = [x * 100 for x in mlp_sweep["analog_noise"]["accuracy_std"]]
        mlp_combined_mean = [x * 100 for x in mlp_sweep["combined"]["accuracy_mean"]]
        mlp_combined_std  = [x * 100 for x in mlp_sweep["combined"]["accuracy_std"]]

        ax.fill_between(noise_pct,
                        np.array(mlp_stuck_mean) - np.array(mlp_stuck_std),
                        np.array(mlp_stuck_mean) + np.array(mlp_stuck_std),
                        alpha=0.10, color="#e6550d")
        ax.plot(noise_pct, mlp_stuck_mean, "^--", color="#e6550d",
                linewidth=1.5, markersize=5, label="MLP — Stuck-at")

        ax.fill_between(noise_pct,
                        np.array(mlp_analog_mean) - np.array(mlp_analog_std),
                        np.array(mlp_analog_mean) + np.array(mlp_analog_std),
                        alpha=0.10, color="#fdae6b")
        ax.plot(noise_pct, mlp_analog_mean, "d--", color="#fdae6b",
                linewidth=1.5, markersize=5, label="MLP — Analog noise")

        ax.fill_between(noise_pct,
                        np.array(mlp_combined_mean) - np.array(mlp_combined_std),
                        np.array(mlp_combined_mean) + np.array(mlp_combined_std),
                        alpha=0.12, color=COLORS["mlp"])
        ax.plot(noise_pct, mlp_combined_mean, "s--", color=COLORS["mlp"],
                linewidth=2.2, markersize=6, label="MLP — Combined", zorder=3)

        mlp_final = mlp_combined_mean[-1]
        delta_mlp = mlp_combined_mean[0] - mlp_combined_mean[-1]
    else:
        # Fallback: single estimated curve (legacy JSON)
        mlp_acc = hw.get("mlp_estimated_accuracy", hw.get("mlp_accuracy", []))
        ax.plot(noise_pct, mlp_acc, "s--", color=COLORS["mlp"], linewidth=2,
                markersize=6, label="MLP — Combined (est.)", zorder=3)
        mlp_final = mlp_acc[-1]
        delta_mlp = mlp_acc[0] - mlp_acc[-1]

    # Chance line
    ax.axhline(20, color=COLORS["chance"], linestyle=":", linewidth=1,
               label="Chance (20%)", zorder=1)

    # Annotations
    delta_hdc = combined_mean[0] - combined_mean[-1]
    ax.annotate(f"HDC: −{delta_hdc:.1f} pp",
                xy=(noise_pct[-1], combined_mean[-1]),
                xytext=(noise_pct[-1] + 1.5, combined_mean[-1] + 6),
                fontsize=9, color=COLORS["hdc"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["hdc"], lw=1.2))
    ax.annotate(f"MLP: −{delta_mlp:.1f} pp",
                xy=(noise_pct[-1], mlp_final),
                xytext=(noise_pct[-1] + 1.5, mlp_final + 6),
                fontsize=9, color=COLORS["mlp"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["mlp"], lw=1.2))

    ax.set_xlabel("Hardware Defect Rate (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hardware Robustness — IMC Defect Tolerance\n"
                 "(Same simulator for both models, 5 trials/level)")
    ymax = max(combined_mean[0], mlp_sweep["combined"]["accuracy_mean"][0] * 100
               if mlp_sweep else 65) + 10
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, noise_pct[-1] + 8)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    if standalone:
        out = os.path.join(FIGURES_DIR, "hardware_robustness.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  [SAVED] {out}")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Energy per Inference  (CPU vs IMC)
# ═══════════════════════════════════════════════════════════════════════════
def plot_energy_comparison(hw, ax=None):
    """Grouped bar chart: energy per inference across compute substrates."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    e = hw["energy_model"]
    hdc_cpu = e["hdc_rff"]["cpu_energy_pj"] / 1e6    # Convert pJ → µJ
    hdc_imc = e["hdc_rff"]["imc_energy_pj"] / 1e6
    mlp_cpu = e["mlp_baseline"]["cpu_energy_pj"] / 1e6
    mlp_imc_total = e["mlp_baseline"]["imc_total_energy_pj"] / 1e6

    labels = ["CPU", "IMC Crossbar"]
    hdc_vals = [hdc_cpu, hdc_imc]
    mlp_vals = [mlp_cpu, mlp_imc_total]

    x = np.arange(len(labels))
    width = 0.3

    bars_hdc = ax.bar(x - width/2, hdc_vals, width, label="HDC-RFF (D=10,000)",
                      color=COLORS["hdc"], edgecolor="white", linewidth=0.5)
    bars_mlp = ax.bar(x + width/2, mlp_vals, width, label="Steel Man MLP",
                      color=COLORS["mlp"], edgecolor="white", linewidth=0.5)

    # Value labels on bars — lift tiny bars so the label is readable
    ymax = max(hdc_cpu, mlp_cpu, hdc_imc, mlp_imc_total)
    label_floor = ymax * 0.08  # minimum height before we offset the label

    for bar in bars_hdc:
        h = bar.get_height()
        label_y = max(h, label_floor) + ymax * 0.01
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f"{h:.2f}" if h < 1 else f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, color=COLORS["hdc"],
                fontweight="bold")
    for bar in bars_mlp:
        h = bar.get_height()
        label_y = max(h, label_floor) + ymax * 0.01
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f"{h:.2f}" if h < 1 else f"{h:.1f}",
                ha="center", va="bottom", fontsize=9, color=COLORS["mlp"],
                fontweight="bold")

    # MLP overhead annotation — placed above bar with arrow since bar is tiny
    overhead_pct = (e["mlp_baseline"]["imc_overhead_pj"] /
                    e["mlp_baseline"]["imc_total_energy_pj"]) * 100
    ax.annotate(f"  {mlp_imc_total:.3f} µJ\n  ({overhead_pct:.0f}% ADC/DAC)",
                xy=(x[1] + width/2, mlp_imc_total),
                xytext=(x[1] + width/2 + 0.25, ymax * 0.30),
                fontsize=8.5, color=COLORS["mlp"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["mlp"], lw=1.2),
                va="center", ha="left")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy per Inference (µJ)")
    ax.set_title("Energy Efficiency — HDC vs MLP\n"
                 "(per inference at 1000 inf/s)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Log scale would hide the story — keep linear but set sensible ylim
    ymax = max(hdc_cpu, mlp_cpu, hdc_imc, mlp_imc_total) * 1.25
    ax.set_ylim(0, ymax)

    if standalone:
        out = os.path.join(FIGURES_DIR, "energy_comparison.png")
        fig.savefig(out)
        plt.close(fig)
        print(f"  [SAVED] {out}")
    return ax


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Combined Dashboard  (2×2)
# ═══════════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(v6, hw):
    """2×2 summary dashboard combining all key results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VICEROY 2026 — HDC for Degraded-Hardware RF Classification",
                 fontsize=15, fontweight="bold", y=0.98)

    # Top-left: Accuracy vs SNR
    if v6:
        plot_accuracy_vs_snr(v6, ax=axes[0, 0])
    else:
        axes[0, 0].text(0.5, 0.5, "V6 results not available",
                        transform=axes[0, 0].transAxes, ha="center")

    # Top-right: Hardware robustness
    if hw:
        plot_hardware_robustness(hw, ax=axes[0, 1])
    else:
        axes[0, 1].text(0.5, 0.5, "Hardware emulation results not available",
                        transform=axes[0, 1].transAxes, ha="center")

    # Bottom-left: Energy comparison
    if hw:
        plot_energy_comparison(hw, ax=axes[1, 0])
    else:
        axes[1, 0].text(0.5, 0.5, "Energy model not available",
                        transform=axes[1, 0].transAxes, ha="center")

    # Bottom-right: Key metrics summary table
    ax_table = axes[1, 1]
    ax_table.axis("off")
    ax_table.set_title("Key Metrics Summary", fontsize=13, pad=15)

    rows = []
    if v6:
        rows.append(["High-SNR Accuracy (HDC)",
                      f"{v6['hdc_rff']['accuracy_high_snr_percent']:.1f}%"])
        rows.append(["High-SNR Accuracy (MLP)",
                      f"{v6['steel_mlp']['accuracy_high_snr_percent']:.1f}%"])
        rows.append(["Training Speedup (HDC vs MLP)",
                      f"{v6['comparison']['training_speedup']:.0f}×"])
    if hw:
        sweep = hw["noise_sweep_detailed"]
        clean = sweep["combined"]["accuracy_mean"][0] * 100
        noisy = sweep["combined"]["accuracy_mean"][-1] * 100
        rows.append(["HDC at 20% Defects", f"{noisy:.1f}%  (−{clean - noisy:.1f} pp)"])

        # MLP at 20% defects — use empirical if available, else estimated
        mlp_sweep = hw.get("mlp_sweep_detailed")
        if mlp_sweep:
            mlp_clean = mlp_sweep["combined"]["accuracy_mean"][0] * 100
            mlp_noisy = mlp_sweep["combined"]["accuracy_mean"][-1] * 100
            rows.append(["MLP at 20% Defects",
                          f"{mlp_noisy:.1f}%  (−{mlp_clean - mlp_noisy:.1f} pp)"])
        else:
            mlp_acc_key = hw.get("mlp_estimated_accuracy", hw.get("mlp_accuracy", []))
            if mlp_acc_key:
                rows.append(["MLP at 20% Defects",
                              f"{mlp_acc_key[-1]:.1f}%  (collapsed)"])

        hdc_batt = hw["energy_model"]["hdc_rff"]["imc_battery_life_hours"]
        rows.append(["HDC Battery Life (10 Wh, IMC)",
                      f"{hdc_batt:,.0f} hrs"])
        mlp_batt = hw["energy_model"]["mlp_baseline"]["cpu_battery_life_hours"]
        rows.append(["MLP Battery Life (10 Wh, CPU)",
                      f"{mlp_batt:,.0f} hrs"])

    if rows:
        table = ax_table.table(
            cellText=rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="left",
            colWidths=[0.55, 0.35],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.6)

        # Style header
        for col in range(2):
            cell = table[0, col]
            cell.set_facecolor("#2d2d2d")
            cell.set_text_props(color="white", fontweight="bold")

        # Alternate row shading
        for row_idx in range(1, len(rows) + 1):
            for col in range(2):
                cell = table[row_idx, col]
                if row_idx % 2 == 0:
                    cell.set_facecolor("#f7f7f7")
                else:
                    cell.set_facecolor("#ffffff")
                cell.set_edgecolor("#dddddd")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGURES_DIR, "viceroy_summary_dashboard.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [SAVED] {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 60)
    print("  VICEROY 2026 — Figure Generator")
    print("=" * 60)

    # Load results
    v6 = load_json(V6_RESULTS)
    hw = load_json(HW_RESULTS)

    if not v6 and not hw:
        print("\n  [ERROR] No result files found. Run benchmarks first:")
        print("    python src/viceroy_hdc_v6_final.py --output results/v6_final_test.json")
        print("    python src/viceroy_hardware_emulation.py")
        sys.exit(1)

    # Individual figures
    if v6:
        print("\n  Generating V6 Accuracy vs SNR ...")
        plot_accuracy_vs_snr(v6)

    if hw:
        print("  Generating Hardware Robustness ...")
        plot_hardware_robustness(hw)
        print("  Generating Energy Comparison ...")
        plot_energy_comparison(hw)

    # Combined dashboard
    print("  Generating Summary Dashboard ...")
    plot_summary_dashboard(v6, hw)

    print("\n" + "=" * 60)
    print("  Done. All figures saved to figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
