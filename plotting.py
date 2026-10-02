"""Figures saved into results/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"


def _bars(ax, labels, series, names, colors):
    x = np.arange(len(labels)); w = 0.8 / len(series)
    for k, (vals, nm, c) in enumerate(zip(series, names, colors)):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, label=nm, color=c)
    ax.set_xticks(x); ax.set_xticklabels(labels)


def generate_plots(results_df, metrics_df):
    os.makedirs(OUT, exist_ok=True)
    m = metrics_df[metrics_df.workload == "LATE_UNSAFE"].reset_index(drop=True)
    cfgs = list(m.config)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(cfgs, m.avg_lead_time, color="skyblue"); ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Average lead time (steps; >0 = earlier than Banker's)")
    ax.set_title("Early-warning lead time (LATE_UNSAFE)"); fig.tight_layout()
    fig.savefig(f"{OUT}/lead_time.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bars(ax, cfgs, [m.fpr, m.fnr], ["False-positive rate", "False-negative rate"], ["#4C72B0", "#DD8452"])
    ax.set_ylabel("Rate"); ax.set_title("Error rates (LATE_UNSAFE)"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/error_rates.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bars(ax, cfgs, [m.detection_rate, m.early_rate],
          ["Flagged at all (1 - FNR)", "Flagged before Banker's"], ["green", "orange"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction of unsafe sequences")
    ax.set_title("Detection vs. early detection (LATE_UNSAFE)"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/detection_rate.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _bars(ax, cfgs, [m.avg_conf_correct, m.avg_conf_wrong],
          ["Avg confidence (correct)", "Avg confidence (wrong)"], ["green", "red"])
    ax.set_ylim(0, 100); ax.set_ylabel("Confidence (%)")
    ax.set_title("Bonus: confidence vs. correctness (LATE_UNSAFE)"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT}/confidence_calibration.png", dpi=150); plt.close(fig)

    # risk probability over time, before/after the shift
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg in cfgs:
        d = results_df[(results_df.config == cfg) & (results_df.workload == "LATE_UNSAFE")]
        curves = np.vstack([c for c in d.curve])
        ax.plot(np.linspace(0, 1, curves.shape[1]), curves.mean(axis=0), label=cfg)
    ax.axvline(0.5, color="k", ls="--", lw=1, label="workload shift")
    ax.set_xlabel("Fraction of trace"); ax.set_ylabel("Mean risk probability")
    ax.set_title("Classifier risk over the trace (LATE_UNSAFE)"); ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(f"{OUT}/risk_over_time.png", dpi=150); plt.close(fig)
