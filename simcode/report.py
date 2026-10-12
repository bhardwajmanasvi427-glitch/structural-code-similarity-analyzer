"""Stage 6 - Similarity prediction, explanation and reporting."""
import os
from collections import Counter

import numpy as np

from .features import (FEATURE_DESCRIPTIONS, METRIC_KEYS, METRIC_LABELS, SIM_FEATURES,
                       analyze, feature_vector, pair_features)
from .training import LABELS, load_model

# Probability-weighted score: each class contributes a representative similarity value.
CLASS_SCORE = {"Low": 10.0, "Moderate": 55.0, "High": 95.0}
HIGH_AT, MODERATE_AT = 75.0, 35.0


def classify(score):
    if score >= HIGH_AT:
        return "High"
    if score >= MODERATE_AT:
        return "Moderate"
    return "Low"


def predict(a, b, model_bundle):
    """Return (score 0-100, category, {label: probability}, feature dict)."""
    pipe = model_bundle["pipeline"]
    vec = np.array([feature_vector(a, b)])
    proba = pipe.predict_proba(vec)[0]
    probs = {LABELS[int(c)]: float(p) for c, p in zip(pipe.classes_, proba)}
    score = sum(probs.get(k, 0.0) * v for k, v in CLASS_SCORE.items())
    return score, classify(score), probs, pair_features(a, b)


def _shared_patterns(a, b, limit=5):
    interesting = ("FuncDef", "For", "While", "DoWhile", "If", "Return", "Call", "Assign", "Switch")
    common = a.sigs & b.sigs
    items = [(s, n) for s, n in common.items() if s.split("(")[0].split(":")[0] in interesting]
    items.sort(key=lambda x: (-len(x[0]), -x[1]))
    return items[:limit]


def explain(a, b, feats):
    """Rule-based, human-readable reasons derived from the compiler-phase information."""
    ma, mb = a.metrics, b.metrics
    same, diff = [], []
    for label, key in (("function(s)", "functions"), ("parameter(s)", "params"),
                       ("loop(s)", "loops"), ("conditional(s)", "conditionals"),
                       ("return statement(s)", "returns"), ("function call(s)", "calls"),
                       ("variable declaration(s)", "decls")):
        if ma[key] == mb[key]:
            if ma[key]:
                same.append("both have %d %s" % (ma[key], label))
        else:
            diff.append("%d vs %d %s" % (ma[key], mb[key], label))
    lines = []
    if same:
        lines.append("Same structure: " + ", ".join(same) + ".")
    if diff:
        lines.append("Different counts: " + ", ".join(diff) + ".")
    la, lb = set(a.loop_kinds), set(b.loop_kinds)
    if la or lb:
        if la == lb:
            lines.append("Both use the same loop construct(s): %s." % ", ".join(sorted(la)))
        else:
            lines.append("Loop constructs differ: A uses %s, B uses %s."
                         % (", ".join(sorted(la)) or "none", ", ".join(sorted(lb)) or "none"))
    if ma["recursive_calls"] or mb["recursive_calls"]:
        lines.append("Recursion: A has %d self-call(s), B has %d." % (ma["recursive_calls"], mb["recursive_calls"]))
    shared_ops = sorted(a.arith_set & b.arith_set)
    if shared_ops:
        lines.append("Shared arithmetic operators: %s." % " ".join(shared_ops))
    lines.append("Max control-flow nesting: A=%d, B=%d; AST depth: A=%d, B=%d."
                 % (ma["ctrl_depth"], mb["ctrl_depth"], ma["ast_depth"], mb["ast_depth"]))
    lines.append("Normalised token sequence similarity %.0f%%, AST sequence similarity %.0f%%."
                 % (100 * feats["token_seq_ratio"], 100 * feats["ast_seq_ratio"]))
    renamed = len(set(a.identifiers) ^ set(b.identifiers))
    if renamed:
        lines.append("Identifier names differ (%d names not shared) - they are ignored by the structural analysis." % renamed)
    return lines


def render_text(a, b, score, category, probs, feats, model_name):
    out = []
    w = out.append
    w("=" * 64)
    w(" SOURCE PROGRAM SIMILARITY ANALYSIS")
    w("=" * 64)
    w("Program A: %s   (%d tokens, %d AST nodes)" % (a.name or "A", a.metrics["tokens"], a.metrics["ast_nodes"]))
    w("Program B: %s   (%d tokens, %d AST nodes)" % (b.name or "B", b.metrics["tokens"], b.metrics["ast_nodes"]))
    w("")
    w("Similarity Score : %s" % category)
    w("Similarity       : %.0f%%" % score)
    w("Model            : %s   (P(Low)=%.2f  P(Moderate)=%.2f  P(High)=%.2f)"
      % (model_name, probs.get("Low", 0), probs.get("Moderate", 0), probs.get("High", 0)))
    w("")
    w("Reason:")
    for line in explain(a, b, feats):
        w("  - " + line)
    w("")
    w("Feature comparison (A | B):")
    for k in METRIC_KEYS:
        mark = " " if a.metrics[k] == b.metrics[k] else "*"
        w("  %s %-30s %5d | %-5d" % (mark, METRIC_LABELS[k], a.metrics[k], b.metrics[k]))
    w("  (* = values differ)")
    w("")
    w("Similarity measures (0 = different, 100 = identical):")
    for k in SIM_FEATURES:
        w("  %-62s %5.1f" % (FEATURE_DESCRIPTIONS[k], 100 * feats[k]))
    pats = _shared_patterns(a, b)
    if pats:
        w("")
        w("Major matching AST patterns (identifier names ignored):")
        for s, n in pats:
            w("  - x%d  %s" % (n, s if len(s) < 90 else s[:87] + "..."))
    w("")
    w("Note: a similarity result is evidence of structural resemblance, not proof of plagiarism.")
    return "\n".join(out)


def plot_comparison(a, b, score, category, feats, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["functions", "params", "loops", "conditionals", "returns", "calls", "decls",
            "assigns", "arith_ops", "rel_ops", "ctrl_depth"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [a.metrics[k] for k in keys], 0.4, label="Program A", color="#3b6ea5")
    ax.bar(x + 0.2, [b.metrics[k] for k in keys], 0.4, label="Program B", color="#e08a3c")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[k].split(" (")[0] for k in keys], rotation=40, ha="right", fontsize=8)
    ax.set_title("Structural feature counts")
    ax.legend()
    ax = axes[1]
    vals = [100 * feats[k] for k in SIM_FEATURES]
    names = [k.replace("_", " ") for k in SIM_FEATURES]
    ax.barh(names[::-1], vals[::-1], color="#5b9a6b")
    ax.set_xlim(0, 100)
    ax.set_title("Similarity measures (%)")
    ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Overall similarity: %.0f%%  (%s)" % (score, category), fontsize=13, fontweight="bold")
    fig.tight_layout()
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
