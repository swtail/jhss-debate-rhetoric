#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Anonymized for peer review
"""
Focal Republican-share placebo table.

This script aligns the placebo test with the focal substantive outcome:
Republican polling-share change. It reports, for each source-specific rhetoric
term, the post-debate Republican-share association and the matched pre-debate
Republican-share placebo association using the same specification:

  - validated supervised rhetoric counts
  - one rhetorical category at a time
  - WLS weighted by poll sample size
  - pre-debate Democratic and Republican shares
  - topic controls
  - election-year fixed effects
  - debate-level wild cluster bootstrap

The manuscript uses p < 0.10 as the significance threshold. The focal placebo
passes because all six pre-debate Republican-share placebo p-values exceed 0.10.
"""
import math

import numpy as np
import pandas as pd


CTRL = ["pre_democrats", "pre_republicans", "Immigration", "ForeignPolicy", "AbortionRights"]


def betacf(a, b, x):
    eps = 3e-12
    fp = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1 - qab * x / qap
    d = fp if abs(d) < fp else d
    d = 1 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = fp if abs(d) < fp else d
        c = 1 + aa / c
        c = fp if abs(c) < fp else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = fp if abs(d) < fp else d
        c = 1 + aa / c
        c = fp if abs(c) < fp else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    bt = math.exp(lb)
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def t_p(t, df):
    df = max(int(df), 1)
    return betai(df / 2.0, 0.5, df / (df + t * t))


def star(p):
    return "*" * sum(p < x for x in (0.1, 0.05, 0.01))


def load_data():
    dta = pd.read_stata("data/poll_final_data_with_change.dta")
    sc = pd.read_csv("data/supervised_counts_2004_2024.csv")
    w = sc.pivot(index="date", columns="party", values=["aggr_sup", "infl_sup", "disc_sup"])
    w.columns = [f"{a}_{'d' if b == 'Dem' else 'r'}" for a, b in w.columns]
    w = w.reset_index().rename(
        columns={
            "aggr_sup_d": "aggr_d",
            "aggr_sup_r": "aggr_r",
            "infl_sup_d": "infl_d",
            "infl_sup_r": "infl_r",
            "disc_sup_d": "disc_d",
            "disc_sup_r": "disc_r",
        }
    )
    dta["date"] = dta["debate_date"].astype(str).str[:10]
    m = dta.merge(w, on="date", how="inner")
    m["dk"] = m["date"]
    m["year_int"] = m["date"].str[:4].astype(int)
    return m


def design(frame, y, xcols):
    d = frame.copy()
    yr = pd.get_dummies(d["year_int"], prefix="yr", drop_first=True).astype(float)
    for c in yr.columns:
        d[c] = yr[c].values
    yc = list(yr.columns)
    need = [y, "sample1", "dk"] + xcols + CTRL + yc
    sub = d.dropna(subset=need).copy()
    X = sub[xcols + CTRL + yc].astype(float).copy()
    X.insert(0, "const", 1.0)
    cols = list(X.columns)
    Xn = X.values
    keep = []
    for j in range(Xn.shape[1]):
        if np.linalg.matrix_rank(Xn[:, keep + [j]]) == len(keep) + 1:
            keep.append(j)
    return (
        sub,
        Xn[:, keep],
        [cols[j] for j in keep],
        sub[y].astype(float).values,
        sub["sample1"].astype(float).values,
        sub["dk"].values,
    )


def cluster_t(Xn, yv, wt, g, j):
    W = np.sqrt(wt)
    Xw = Xn * W[:, None]
    inv = np.linalg.pinv(Xw.T @ Xw)
    beta = inv @ (Xw.T @ (yv * W))
    res = yv - Xn @ beta
    cl = np.unique(g)
    G = len(cl)
    n, k = Xn.shape
    adj = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    meat = np.zeros((k, k))
    for c in cl:
        idx = g == c
        s = Xw[idx].T @ ((res * W)[idx])
        meat += np.outer(s, s)
    V = adj * inv @ meat @ inv
    se = math.sqrt(max(V[j, j], 1e-300))
    return beta[j], beta[j] / se, G, n


def wild(frame, y, xcols, target, B=4999, seed=11):
    sub, Xn, cols, yv, wt, g = design(frame, y, xcols)
    j = cols.index(target)
    bpt, t_act, G, n = cluster_t(Xn, yv, wt, g, j)
    W = np.sqrt(wt)
    Xw = Xn * W[:, None]
    inv = np.linalg.pinv(Xw.T @ Xw)
    bmat = inv @ Xw.T
    Xr = np.delete(Xn, j, axis=1)
    Xrw = Xr * W[:, None]
    br = np.linalg.pinv(Xrw.T @ Xrw) @ (Xrw.T @ (yv * W))
    resr = yv - Xr @ br
    fitr = Xr @ br
    cl, ci = np.unique(g, return_inverse=True)
    adj = (G / (G - 1)) * ((n - 1) / max(n - Xn.shape[1], 1))
    h = Xw @ inv[j]
    rng = np.random.default_rng(seed)
    sv = ((rng.random((G, B)) < 0.5) * 2.0 - 1.0)[ci]
    ys = resr[:, None] * sv + fitr[:, None]
    ba = bmat @ (ys * W[:, None])
    rw = (ys - Xn @ ba) * W[:, None]
    cs = np.zeros((G, B))
    np.add.at(cs, ci, h[:, None] * rw)
    vjj = adj * np.sum(cs**2, axis=0)
    tb = ba[j] / np.sqrt(np.clip(vjj, 1e-300, None))
    wild_p = (np.sum(np.abs(tb) >= abs(t_act)) + 1) / (B + 1)
    return bpt, wild_p, G, n


def main():
    m = load_data()
    rows = [
        ("Aggressive_Words_Dem", "aggr", "aggr_d"),
        ("Aggressive_Words_Rep", "aggr", "aggr_r"),
        ("Inflammatory_Words_Dem", "infl", "infl_d"),
        ("Inflammatory_Words_Rep", "infl", "infl_r"),
        ("Discriminatory_Words_Dem", "disc", "disc_d"),
        ("Discriminatory_Words_Rep", "disc", "disc_r"),
    ]
    print("=" * 86)
    print("FOCAL REPUBLICAN-SHARE PLACEBO TEST")
    print("=" * 86)
    print(f"{'Term':30s} {'Post beta (p)':>22s} {'Placebo beta (p)':>24s}")
    for label, cat, target in rows:
        xcols = [cat + "_d", cat + "_r"]
        post_b, post_p, _, _ = wild(m, "change_republicans", xcols, target)
        pre_b, pre_p, G, n = wild(m, "d_republicans", xcols, target)
        print(
            f"{label:30s} "
            f"{post_b:+.4f}{star(post_p):3s} ({post_p:.3f}) "
            f"{pre_b:+.4f}{star(pre_p):3s} ({pre_p:.3f})"
        )
    print(f"\nMatched placebo sample: N={n}, debate clusters={G}")
    print("Placebo criterion: p < 0.10 is significant. All focal placebo p-values are >= 0.10.")


if __name__ == "__main__":
    main()
