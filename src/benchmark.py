#!/usr/bin/env python3
"""
Benchmark naive_dp (V1) vs run_dp_v2 (V2) across all four synthetic datasets.

Ground truth reconstruction
---------------------------
  clean / big_indel : GT coord pairs are (i, i) for i in 0..n_coords-1
                      (both have |A| == |B|, same-index sites correspond)
  missing_sites     : B ⊂ A by coordinate value → match on exact value
  extra_sites       : A ⊂ B by coordinate value → match on exact value

Evaluation is done at the **coordinate-pair** level (not interval level):
  a predicted interval match (i_int, j_int) implies
      A_coord[i_int] ↔ B_coord[j_int]  and  A_coord[i_int+1] ↔ B_coord[j_int+1]

Metrics
-------
  F1 / Precision / Recall  — how well the algorithm recovers the true site pairings
  Mean positional error (bp) on true-positive pairs
  Fraction of TP pairs within eps bp
  Runtime (seconds)
"""

import os
import sys
import json
import time

import numpy as np

# allow direct import from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_dp import load_1d, intervals_from_coords, run_dp
from run_dp_v2 import run_dp_v2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SRC_DIR, '..', 'data', 'synthetic')
DATASETS  = ['clean', 'missing_sites', 'extra_sites', 'big_indel']

# ---------------------------------------------------------------------------
# Shared hyper-parameters (keep defaults from each script)
# ---------------------------------------------------------------------------
ALPHA       = 100.0
BETA        = 1.0     # V1 only
SIGMA       = 1.0     # V2 only
GAP_PENALTY = 80.0    # V1 only
GAMMA       = 10.0    # V2 only
LAMBDA_A    = 2.0     # V2 only
LAMBDA_B    = 2.0     # V2 only
EPS         = 50

# ---------------------------------------------------------------------------
# Ground-truth construction
# ---------------------------------------------------------------------------

def build_ground_truth(name, A_coords, B_coords):
    """
    Returns a frozenset of (A_coord_idx, B_coord_idx) true-positive pairs.
    """
    if name in ('clean', 'big_indel'):
        # same number of sites; correspondence is by index
        n = min(len(A_coords), len(B_coords))
        return frozenset((i, i) for i in range(n))

    elif name == 'missing_sites':
        # B is a strict subset of A (sites deleted); match by coordinate value
        a_val_to_idx = {int(v): i for i, v in enumerate(A_coords)}
        pairs = set()
        for j, v in enumerate(B_coords):
            i = a_val_to_idx.get(int(v))
            if i is not None:
                pairs.add((i, j))
        return frozenset(pairs)

    elif name == 'extra_sites':
        # A is a strict subset of B (sites inserted); match by coordinate value
        b_val_to_idx = {int(v): j for j, v in enumerate(B_coords)}
        pairs = set()
        for i, v in enumerate(A_coords):
            j = b_val_to_idx.get(int(v))
            if j is not None:
                pairs.add((i, j))
        return frozenset(pairs)

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Convert backtrack paths → predicted coordinate pairs
# ---------------------------------------------------------------------------

def anchors_from_v1_path(path, n_A_coords, n_B_coords):
    """
    V1 MATCH on interval index (i, j) implies:
        A_coord[i]   ↔ B_coord[j]
        A_coord[i+1] ↔ B_coord[j+1]
    """
    pairs = set()
    for op, i, j in path:
        if op == 'MATCH':
            for ai, bj in [(i, j), (i + 1, j + 1)]:
                if 0 <= ai < n_A_coords and 0 <= bj < n_B_coords:
                    pairs.add((ai, bj))
    return frozenset(pairs)


def anchors_from_v2_path(path, n_A_coords, n_B_coords):
    """
    V2 move (prev=(pi,pj), curr=(ci,cj)) anchors boundary coord indices:
        A_coord[pi] ↔ B_coord[pj]   (left boundary)
        A_coord[ci] ↔ B_coord[cj]   (right boundary)
    Gap moves (GAP_A / GAP_B) consume only one side and do not anchor pairs.
    """
    pairs = set()
    for move in path:
        if move.get('type') in ('GAP_A', 'GAP_B'):
            continue
        pi, pj = move['prev']
        ci, cj = move['curr']
        for ai, bj in [(pi, pj), (ci, cj)]:
            if 0 <= ai < n_A_coords and 0 <= bj < n_B_coords:
                pairs.add((ai, bj))
    return frozenset(pairs)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predicted, gt, A_coords, B_coords, eps):
    tp = predicted & gt
    fp = predicted - gt
    fn = gt - predicted

    precision = len(tp) / len(predicted) if predicted else 0.0
    recall    = len(tp) / len(gt)        if gt        else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    errors = [abs(int(A_coords[ai]) - int(B_coords[bj])) for ai, bj in tp]
    mean_err  = float(np.mean(errors))                        if errors else float('nan')
    frac_eps  = float(np.mean(np.array(errors) <= eps))       if errors else float('nan')

    return {
        'precision':            round(precision, 4),
        'recall':               round(recall, 4),
        'f1':                   round(f1, 4),
        'n_tp':                 len(tp),
        'n_fp':                 len(fp),
        'n_fn':                 len(fn),
        'n_gt':                 len(gt),
        'n_pred':               len(predicted),
        'mean_pos_error_tp_bp': round(mean_err, 2)  if not np.isnan(mean_err)  else None,
        'frac_within_eps_tp':   round(frac_eps, 4)  if not np.isnan(frac_eps)  else None,
    }


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(eps=EPS, save_json=True):
    all_results = {}

    for name in DATASETS:
        ddir     = os.path.join(DATA_ROOT, name)
        A_coords = load_1d(os.path.join(ddir, 'A_coords.csv'))
        B_coords = load_1d(os.path.join(ddir, 'B_coords.csv'))
        A_int    = intervals_from_coords(A_coords)
        B_int    = intervals_from_coords(B_coords)
        gt       = build_ground_truth(name, A_coords, B_coords)

        print(f"\n[{name}]  A={len(A_coords)} sites  B={len(B_coords)} sites  "
              f"GT pairs={len(gt)}", flush=True)

        # ---- V1 ----
        t0 = time.perf_counter()
        v1_score, v1_path = run_dp(
            A_int, B_int, ALPHA, BETA, GAP_PENALTY, eps, hard_eps=False
        )
        v1_time = time.perf_counter() - t0

        v1_pred    = anchors_from_v1_path(v1_path, len(A_coords), len(B_coords))
        v1_metrics = compute_metrics(v1_pred, gt, A_coords, B_coords, eps)
        v1_metrics['score']      = round(float(v1_score), 2)
        v1_metrics['runtime_s']  = round(v1_time, 3)
        print(f"  V1 done in {v1_time:.1f}s  score={v1_score:.1f}  F1={v1_metrics['f1']:.4f}", flush=True)

        # ---- V2 ----
        t0 = time.perf_counter()
        v2_score, v2_path = run_dp_v2(
            A_int, B_int, ALPHA, SIGMA, GAMMA, LAMBDA_A, LAMBDA_B, eps,
            gap_penalty=GAP_PENALTY
        )
        v2_time = time.perf_counter() - t0

        v2_pred    = anchors_from_v2_path(v2_path, len(A_coords), len(B_coords))
        v2_metrics = compute_metrics(v2_pred, gt, A_coords, B_coords, eps)
        v2_metrics['score']      = round(float(v2_score), 2)
        v2_metrics['runtime_s']  = round(v2_time, 3)
        print(f"  V2 done in {v2_time:.1f}s  score={v2_score:.1f}  F1={v2_metrics['f1']:.4f}", flush=True)

        all_results[name] = {
            'n_A': len(A_coords),
            'n_B': len(B_coords),
            'n_gt_pairs': len(gt),
            'V1': v1_metrics,
            'V2': v2_metrics,
        }

    _print_table(all_results)

    if save_json:
        out_path = os.path.join(SRC_DIR, '..', 'benchmark_results.json')
        out_path = os.path.normpath(out_path)
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to: {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

COLS = [
    ('f1',                   'F1'),
    ('precision',            'Precision'),
    ('recall',               'Recall'),
    ('n_tp',                 'TP pairs'),
    ('n_fp',                 'FP pairs'),
    ('n_fn',                 'FN pairs'),
    ('mean_pos_error_tp_bp', 'Mean Err (bp)'),
    ('frac_within_eps_tp',   'Frac w/in eps'),
    ('runtime_s',            'Runtime (s)'),
]

def _fmt(v):
    if v is None:           return f"{'N/A':>14}"
    if isinstance(v, int):  return f"{v:>14,}"
    return f"{v:>14.4f}"

def _print_table(results):
    col_labels = [lbl for _, lbl in COLS]
    header = f"{'Dataset':<18} {'Alg':<4}  " + "  ".join(f"{c:>14}" for c in col_labels)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name in DATASETS:
        r = results[name]
        for alg in ('V1', 'V2'):
            m    = r[alg]
            vals = "  ".join(_fmt(m.get(k)) for k, _ in COLS)
            print(f"{name:<18} {alg:<4}  {vals}")
        print(sep)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_benchmark()
