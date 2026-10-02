"""Runs the experiments: train on RANDOM, test on RANDOM (no shift) and LATE_UNSAFE (shift)."""
import os
import numpy as np
import pandas as pd
from .workload import generate_sequence, label
from .ml import train, risk_curve, sequence_features, explain

RESULTS_DIR = "results"
THRESHOLD = 0.5
SEED = 42


def evaluate(clf, seqs, workload):
    rows = []
    for s in seqs:
        unsafe, first_unsafe = label(s)
        p = risk_curve(clf, s)
        flagged_steps = np.where(p >= THRESHOLD)[0]
        first_flag = int(flagged_steps[0]) if len(flagged_steps) else None
        conf = np.maximum(p, 1 - p)
        correct = (p >= THRESHOLD) == unsafe          # step-level correctness
        shift = s["shift_step"]
        rows.append({
            "workload": workload,
            "unsafe": unsafe,
            "first_unsafe_step": first_unsafe,
            "first_flag_step": first_flag,
            "lead_time": (first_unsafe - first_flag) if (unsafe and first_flag is not None) else np.nan,
            "flagged": first_flag is not None,
            "conf_correct": float(conf[correct].mean()) if correct.any() else np.nan,
            "conf_wrong": float(conf[~correct].mean()) if (~correct).any() else np.nan,
            "risk_pre_shift": float(p[:shift].mean()) if shift else np.nan,
            "risk_post_shift": float(p[shift:].mean()) if shift else np.nan,
            "curve": p,
        })
    return rows


def summarise(df):
    uns, saf = df[df.unsafe], df[~df.unsafe]
    return {
        "n_unsafe": len(uns), "n_safe": len(saf),
        "avg_lead_time": uns.lead_time.mean(),                       # flagged unsafe seqs only
        "detection_rate": uns.flagged.mean() if len(uns) else np.nan,     # = 1 - FNR
        "early_rate": (uns.lead_time > 0).mean() if len(uns) else np.nan,  # flagged strictly before Banker's
        "fpr": saf.flagged.mean() if len(saf) else np.nan,
        "fnr": 1 - uns.flagged.mean() if len(uns) else np.nan,
        "avg_conf_correct": df.conf_correct.mean() * 100,
        "avg_conf_wrong": df.conf_wrong.mean() * 100,
        "risk_pre_shift": df.risk_pre_shift.mean(),
        "risk_post_shift": df.risk_post_shift.mean(),
    }


def run_experiment_configs(configs, n_train=1000, n_test=300):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_rows, metrics = [], []
    explanations = []
    for cfg in configs:
        n, m = cfg["processes"], cfg["resources"]
        train_seqs = [generate_sequence(rng, n, m, "RANDOM") for _ in range(n_train)]
        clf = train(train_seqs, max_depth=4, seed=SEED)
        for workload in ("RANDOM", "LATE_UNSAFE"):
            test_seqs = [generate_sequence(rng, n, m, workload) for _ in range(n_test)]
            rows = evaluate(clf, test_seqs, workload)
            df = pd.DataFrame(rows)
            df["config"] = cfg["name"]
            all_rows.append(df)
            metrics.append({"config": cfg["name"], "processes": n, "resources": m,
                            "workload": workload, **summarise(df)})
            if workload == "LATE_UNSAFE":                       # bonus: sample explanations
                s = next((x for x in test_seqs if label(x)[0]), test_seqs[0])
                f, p = sequence_features(s), risk_curve(clf, s)
                for t in range(0, len(p), max(1, len(p) // 6)):
                    explanations.append(f"{cfg['name']} step {t:2d}: {explain(f[t], p[t])}"
                                        f"  [Banker's: {'safe' if s['safe_flags'][t] else 'UNSAFE'}]")
    with open(os.path.join(RESULTS_DIR, "explanations.txt"), "w") as fh:
        fh.write("\n".join(explanations) + "\n")
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(metrics)


def generate_summaries(metrics_df):
    cols = ["config", "processes", "resources", "workload", "n_safe", "n_unsafe", "avg_lead_time",
            "detection_rate", "early_rate", "fpr", "fnr", "avg_conf_correct", "avg_conf_wrong",
            "risk_pre_shift", "risk_post_shift"]
    out = metrics_df[cols].round(3)
    out.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "metrics_summary.md"), "w") as fh:
        fh.write(out.to_markdown(index=False) if _has_tabulate() else out.to_string(index=False))
    print(out.to_string(index=False))


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
