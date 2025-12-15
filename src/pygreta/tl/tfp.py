from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as sts

from pygreta.tl._utils import f_beta_score


def _compute_overlap_pval(
    grn: pd.DataFrame,
    tf_a: str,
    tf_b: str,
) -> tuple:
    trg_a = set(grn[grn["source"] == tf_a]["target"])
    trg_b = set(grn[grn["source"] == tf_b]["target"])
    total = set(grn["target"])
    a = len(trg_a & trg_b)
    if a > 0:
        b = len(trg_a - trg_b)
        c = len(trg_b - trg_a)
        d = len(total - (trg_a | trg_b))
        s, p = sts.fisher_exact([[a, b], [c, d]], alternative="greater")
    else:
        s, p = 0, np.nan
    return s, p


def _find_pairs(
    grn: pd.DataFrame,
    thr_pval: float,
) -> set:
    df = []
    for tf_a, tf_b in combinations(grn["source"].unique(), r=2):
        s, p = _compute_overlap_pval(grn=grn, tf_a=tf_a, tf_b=tf_b)
        df.append([tf_a, tf_b, s, p])
    df = pd.DataFrame(df, columns=["tf_a", "tf_b", "stat", "pval"]).dropna()
    if df.shape[0] > 0:
        df["padj"] = sts.false_discovery_control(df["pval"], method="bh")
        df = df[df["padj"] < thr_pval]
        pairs = {"|".join(sorted([a, b])) for a, b in zip(df["tf_a"], df["tf_b"], strict=True)}
    else:
        pairs = set()
    return pairs


def _tfp(
    grn: pd.DataFrame,
    db: pd.DataFrame,
    thr_pval: float = 0.05,
) -> tuple:
    grn = grn.drop_duplicates(["source", "target"])
    tfs = set(db[0]) | set(db[1])
    grn = grn[grn["source"].isin(tfs)]
    db = {"|".join(sorted([a, b])) for a, b in zip(db[0], db[1], strict=True)}
    if grn.shape[0] > 1:  # Need at least 2 TFs in grn
        # Find pairs
        p_grn = _find_pairs(grn=grn, thr_pval=thr_pval)
        # Compute F score
        tp = len(p_grn & db)
        if tp > 0:
            fp = len(p_grn - db)
            fn = len(db - p_grn)
            rcl = tp / (tp + fn)
            prc = tp / (tp + fp)
            f01 = f_beta_score(prc, rcl)
        else:
            prc, rcl, f01 = 0.0, 0.0, 0.0
    else:
        prc, rcl, f01 = np.nan, np.nan, np.nan
    return prc, rcl, f01
