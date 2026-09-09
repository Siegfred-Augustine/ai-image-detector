"""
training/metrics.py

Evaluation and statistical-testing utilities, per Handoff Sections 16-17:

    - Precision, Recall, F1, and the binary confusion matrix, with the
      SAME orientation the doc uses (positive class = AI-generated = 1):

                            Actual
                         AI       Real
        Pred AI          TP        FP
        Pred Real        FN        TN

      The doc explicitly flags FN (an AI image accepted as real) as the
      most dangerous forensic failure mode, so it's surfaced as its own
      field everywhere rather than only being buried inside a matrix.

    - A paired t-test across 5 seeded runs' per-metric values (Section 17).

    - McNemar's test on two models' per-image predictions on the SAME
      test set (Section 17) -- appropriate here specifically because
      Section 17 also specifies that compared models are evaluated on
      the same test images (image-level prediction agreement/disagreement).

Kept dependency-light: only numpy + scipy.stats (already required for
scikit-learn), no statsmodels dependency for McNemar's test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

# Positive class, per Handoff Section 1: 0 = Real, 1 = AI-generated.
POSITIVE_LABEL = 1


@dataclass
class ClassificationMetrics:
    """Container for one evaluation run's headline numbers."""

    precision: float
    recall: float
    f1: float
    tp: int  # AI correctly classified as AI
    tn: int  # Real correctly classified as Real
    fp: int  # Real incorrectly classified as AI
    fn: int  # AI incorrectly classified as Real  -- the "dangerous" failure (Section 16)
    accuracy: float
    n: int  # total number of evaluated samples

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "n": self.n,
            "fn_rate": self.fn / self.n if self.n else 0.0,  # dangerous-failure rate, Section 16
        }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> ClassificationMetrics:
    """
    Compute precision/recall/F1/confusion matrix for one run.

    Args:
        y_true: ground-truth labels, 0 = Real, 1 = AI-generated.
        y_pred: predicted labels, same convention.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must be the same length"

    precision = precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)

    # sklearn's confusion_matrix with labels=[1, 0] gives us the exact
    # orientation from Handoff Section 16 directly:
    #   row 0 = predicted AI,   row 1 = predicted Real
    #   col 0 = actual AI,      col 1 = actual Real
    cm = sk_confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tn = int(cm[1, 0]), int(cm[1, 1])

    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=float(accuracy),
        n=len(y_true),
    )


@dataclass
class RunResult:
    """One seeded training run's evaluation result, for statistical aggregation."""

    seed: int
    metrics: ClassificationMetrics
    y_true: List[int] = field(repr=False)
    y_pred: List[int] = field(repr=False)


@dataclass
class AggregateStats:
    """Mean/std of a metric across the Section-17-mandated 5 independent runs."""

    mean: float
    std: float
    values: List[float]

    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std, "values": self.values}


def aggregate_metric(runs: Sequence[RunResult], metric_name: str) -> AggregateStats:
    """
    Mean and (sample, ddof=1) standard deviation of one metric
    (precision / recall / f1 / accuracy) across multiple seeded runs.
    """
    values = [getattr(r.metrics, metric_name) for r in runs]
    values_arr = np.asarray(values, dtype=np.float64)
    mean = float(values_arr.mean())
    std = float(values_arr.std(ddof=1)) if len(values_arr) > 1 else 0.0
    return AggregateStats(mean=mean, std=std, values=values)


@dataclass
class PairedTTestResult:
    metric: str
    t_statistic: float
    degrees_of_freedom: int
    p_value: float
    significant: bool  # at alpha = 0.05, per Handoff Section 17
    mean_a: float
    mean_b: float


def paired_ttest(
    runs_a: Sequence[RunResult],
    runs_b: Sequence[RunResult],
    metric_name: str,
    alpha: float = 0.05,
) -> PairedTTestResult:
    """
    Paired t-test comparing corresponding metric values across N
    independent runs of two model configurations (Handoff Section 17).

    "Corresponding" means the i-th run of model A and the i-th run of
    model B should share the same random seed (same data ordering /
    initialization scheme), which is exactly how
    experiments/run_experiments.py pairs them up.

    Args:
        runs_a, runs_b: RunResult lists for the two models being
            compared, same length, seed-aligned (runs_a[i].seed ==
            runs_b[i].seed is the expected/checked invariant).
        metric_name: one of "precision", "recall", "f1", "accuracy".
        alpha: significance threshold (Handoff Section 17: alpha = 0.05).
    """
    assert len(runs_a) == len(runs_b), "Paired t-test requires equal-length, seed-aligned run lists"
    seeds_a = [r.seed for r in runs_a]
    seeds_b = [r.seed for r in runs_b]
    if seeds_a != seeds_b:
        raise ValueError(
            f"Run seeds are not aligned between the two models being compared: "
            f"{seeds_a} vs {seeds_b}. Paired tests require matched seeds."
        )

    values_a = np.array([getattr(r.metrics, metric_name) for r in runs_a], dtype=np.float64)
    values_b = np.array([getattr(r.metrics, metric_name) for r in runs_b], dtype=np.float64)

    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    dof = len(values_a) - 1

    return PairedTTestResult(
        metric=metric_name,
        t_statistic=float(t_stat),
        degrees_of_freedom=dof,
        p_value=float(p_value),
        significant=bool(p_value < alpha),
        mean_a=float(values_a.mean()),
        mean_b=float(values_b.mean()),
    )


@dataclass
class McNemarResult:
    statistic: float
    p_value: float
    significant: bool  # at alpha = 0.05, per Handoff Section 17
    n01: int  # model A wrong, model B right
    n10: int  # model A right, model B wrong
    exact: bool  # whether the exact binomial test was used (small discordant count)


def mcnemar_test(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    alpha: float = 0.05,
) -> McNemarResult:
    """
    McNemar's test comparing two models' per-image predictions on the
    SAME test set (Handoff Section 17): "Applied because compared
    models use the same test images. It compares image-level prediction
    disagreements."

    Uses the discordant-pair counts:
        n10 = # images model A got right and model B got wrong
        n01 = # images model A got wrong and model B got right

    For small discordant counts (n01 + n10 < 25) this uses the exact
    two-sided binomial test (recommended when the chi-square
    approximation is unreliable); otherwise the standard
    continuity-corrected chi-square statistic:

        statistic = (|n01 - n10| - 1)^2 / (n01 + n10)

    is used, compared against a chi-square distribution with 1 degree
    of freedom. This mirrors the common statsmodels
    `mcnemar(..., exact=None)` behavior without adding that dependency.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    assert y_true.shape == pred_a.shape == pred_b.shape, "y_true, pred_a, pred_b must be the same length"

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    n10 = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n01 = int(np.sum(~correct_a & correct_b))  # A wrong, B right
    n_discordant = n01 + n10

    if n_discordant == 0:
        # Models agree on every single prediction -- no evidence of a
        # difference; report p = 1.0 rather than dividing by zero.
        return McNemarResult(statistic=0.0, p_value=1.0, significant=False, n01=n01, n10=n10, exact=True)

    if n_discordant < 25:
        # Exact two-sided binomial test: under H0, n10 ~ Binomial(n_discordant, 0.5).
        result = stats.binomtest(min(n10, n01), n=n_discordant, p=0.5, alternative="two-sided")
        p_value = float(result.pvalue)
        statistic = float(min(n10, n01))
        exact = True
    else:
        statistic = float((abs(n01 - n10) - 1) ** 2) / n_discordant
        p_value = float(1 - stats.chi2.cdf(statistic, df=1))
        exact = False

    return McNemarResult(
        statistic=statistic,
        p_value=p_value,
        significant=bool(p_value < alpha),
        n01=n01,
        n10=n10,
        exact=exact,
    )


def format_confusion_matrix(metrics: ClassificationMetrics) -> str:
    """Render the confusion matrix in the exact layout from Handoff Section 16."""
    return (
        "                    Actual\n"
        "                 AI       Real\n"
        f"Pred AI      {metrics.tp:>8d} {metrics.fp:>8d}\n"
        f"Pred Real    {metrics.fn:>8d} {metrics.tn:>8d}\n"
    )
