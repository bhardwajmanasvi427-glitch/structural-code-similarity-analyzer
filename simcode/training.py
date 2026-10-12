"""Dataset construction, model comparison, selection and saving."""
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, analyze, feature_vector
from .programs import HOLDOUT, TRAIN_FAMILIES, generate_pairs

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "similarity_model.joblib")
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")
LABELS = ["Low", "Moderate", "High"]


def build_frame(pairs):
    rows = []
    for sa, sb, label, fa, fb in pairs:
        a, b = analyze(sa), analyze(sb)
        rows.append(feature_vector(a, b) + [label, fa, fb])
    return pd.DataFrame(rows, columns=FEATURE_NAMES + ["label", "family_a", "family_b"])


def _candidates():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    def pipe(clf):
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])

    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=5000, C=1.0)),
        "DecisionTree": pipe(DecisionTreeClassifier(max_depth=10, random_state=0)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
        "SVM(RBF)": pipe(SVC(kernel="rbf", C=3.0, probability=True, random_state=0)),
    }


def _plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_importance(pipe, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(axis=0)
    else:
        return False
    order = np.argsort(imp)[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([FEATURE_NAMES[i] for i in order], imp[order], color="#3b6ea5")
    ax.set_title("Top features used by the selected model", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def run_training(n_pairs=3600, n_unseen=600, seed=7, verbose=True, save=True):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    def log(*a):
        if verbose:
            print(*a, flush=True)

    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    log("[1/5] Generating %d labelled training pairs (+%d unseen-algorithm test pairs)..."
        % (n_pairs, n_unseen))
    seen = build_frame(generate_pairs(n_pairs, seed, TRAIN_FAMILIES))
    unseen = build_frame(generate_pairs(n_unseen, seed + 1, TRAIN_FAMILIES, must_include=HOLDOUT))
    log("      parsed + featurised in %.1fs; held-out algorithms: %s" % (time.time() - t0, ", ".join(HOLDOUT)))

    X, y = seen[FEATURE_NAMES].values, seen["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xun, yun = unseen[FEATURE_NAMES].values, unseen["label"].values

    if save:
        seen.assign(split="seen").to_csv(os.path.join(DATA_DIR, "pair_features_seen.csv"), index=False)
        unseen.assign(split="unseen").to_csv(os.path.join(DATA_DIR, "pair_features_unseen.csv"), index=False)

    log("[2/5] Cross-validating candidate models (5-fold, macro-F1)...")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows, models = [], _candidates()
    for name, model in models.items():
        t1 = time.time()
        scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="f1_macro")
        model.fit(Xtr, ytr)
        rows.append({
            "model": name,
            "cv_f1_macro": scores.mean(),
            "cv_std": scores.std(),
            "test_seen_acc": accuracy_score(yte, model.predict(Xte)),
            "test_unseen_acc": accuracy_score(yun, model.predict(Xun)),
            "test_unseen_f1": f1_score(yun, model.predict(Xun), average="macro"),
        })
        log("      %-20s CV F1 %.3f (+/- %.3f)  seen-test acc %.3f  unseen-test acc %.3f  [%.1fs]"
            % (name, rows[-1]["cv_f1_macro"], rows[-1]["cv_std"], rows[-1]["test_seen_acc"],
               rows[-1]["test_unseen_acc"], time.time() - t1))
    table = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False)

    best_name = table.iloc[0]["model"]
    best = models[best_name]
    log("[3/5] Selected model (highest cross-validated macro-F1): %s" % best_name)

    log("[4/5] Evaluating selected model...")
    report_lines = ["Selected model: %s" % best_name, ""]
    for tag, (Xs, ys) in (("SEEN-algorithm test set (20%% split, %d pairs)" % len(yte), (Xte, yte)),
                         ("UNSEEN-algorithm test set (%d pairs, held-out: %s)" % (len(yun), ", ".join(HOLDOUT)),
                          (Xun, yun))):
        pred = best.predict(Xs)
        cm = confusion_matrix(ys, pred, labels=[0, 1, 2])
        report_lines += [tag, "accuracy = %.3f" % accuracy_score(ys, pred),
                         classification_report(ys, pred, labels=[0, 1, 2], target_names=LABELS, digits=3),
                         "confusion matrix (rows=actual Low/Moderate/High):", str(cm), ""]
        if save:
            fname = "confusion_seen.png" if tag.startswith("SEEN") else "confusion_unseen.png"
            _plot_confusion(cm, tag.split(" (")[0], os.path.join(RESULTS_DIR, fname))
    report = "\n".join(report_lines)
    log(report)

    if save:
        log("[5/5] Saving model and results...")
        table.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as fh:
            fh.write(table.to_string(index=False) + "\n\n" + report + "\n")
        _plot_importance(best, os.path.join(RESULTS_DIR, "feature_importance.png"))
        joblib.dump({"pipeline": best, "model_name": best_name, "feature_names": FEATURE_NAMES,
                     "labels": LABELS, "comparison": table.to_dict("records")}, MODEL_PATH)
        log("      model -> %s" % MODEL_PATH)
    log("Done in %.1fs." % (time.time() - t0))
    return table, best_name


def load_model(path=None, auto_train=True):
    path = path or MODEL_PATH
    if not os.path.exists(path):
        if not auto_train:
            raise FileNotFoundError("model not found at %s - run: python train.py" % path)
        print("No trained model found - training one now (about a minute)...", flush=True)
        run_training()
    return joblib.load(path)
