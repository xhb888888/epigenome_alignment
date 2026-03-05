#!/usr/bin/env python3
"""
Run V1 and/or V2 on the real E. coli data and compute proxy evaluation metrics.

Since there is no ground truth for real data, evaluation uses:
  - Match rate          : fraction of sites successfully paired (not gapped)
  - Error distribution  : statistics on |distA - distB| for matched pairs
  - Gap count & density : how many / where gaps fall along the genome
  - Implied coord map   : (A_coord, B_coord) scatter for all anchor pairs
                          (saved as TSV for external dotplot visualization)

Usage
-----
  python run_real.py --alg v1          # run V1 only (~10 min)
  python run_real.py --alg v2          # run V2 only (~5 hr)
  python run_real.py --alg both        # run both and compute agreement
"""

import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_dp import load_1d, intervals_from_coords, run_dp, make_anchor_pairs
from run_dp_v2 import run_dp_v2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR  = os.path.normpath(os.path.join(SRC_DIR, '..'))
DATA_DIR  = os.path.join(PROJ_DIR, 'data', 'real', 'ecoli_realdata_synth_format')
OUT_ROOT  = os.path.join(PROJ_DIR, 'results', 'real')

# ---------------------------------------------------------------------------
# Default parameters (same as benchmark)
# ---------------------------------------------------------------------------
ALPHA       = 100.0
BETA        = 1.0
SIGMA       = 1.0
GAP_PENALTY = 80.0
GAMMA       = 10.0
LAMBDA_A    = 2.0
LAMBDA_B    = 2.0
EPS         = 50


# ---------------------------------------------------------------------------
# Proxy metrics (no ground truth)
# ---------------------------------------------------------------------------

def compute_proxy_metrics(n_A_sites, n_B_sites, n_matched, n_gap_a, n_gap_b,
                          errors, eps, runtime):
    """
    errors : list of |distA - distB| for each matched/merged step
    """
    match_rate_A = n_matched / n_A_sites if n_A_sites else 0
    match_rate_B = n_matched / n_B_sites if n_B_sites else 0

    metrics = {
        'n_A_sites'     : int(n_A_sites),
        'n_B_sites'     : int(n_B_sites),
        'n_matched'     : int(n_matched),
        'n_gap_a'       : int(n_gap_a),
        'n_gap_b'       : int(n_gap_b),
        'match_rate_A'  : round(match_rate_A, 4),
        'match_rate_B'  : round(match_rate_B, 4),
        'runtime_s'     : round(runtime, 2),
    }

    if errors:
        arr = np.array(errors)
        metrics.update({
            'error_mean_bp'     : round(float(arr.mean()), 2),
            'error_median_bp'   : round(float(np.median(arr)), 2),
            'error_std_bp'      : round(float(arr.std()), 2),
            'error_p90_bp'      : round(float(np.percentile(arr, 90)), 2),
            'error_p99_bp'      : round(float(np.percentile(arr, 99)), 2),
            'frac_within_eps'   : round(float((arr <= eps).mean()), 4),
            'frac_exact_match'  : round(float((arr == 0).mean()), 4),
        })
    return metrics


# ---------------------------------------------------------------------------
# V1 runner
# ---------------------------------------------------------------------------

def run_v1(A_coords, B_coords, A_int, B_int, outdir, eps):
    os.makedirs(outdir, exist_ok=True)
    print(f"[V1] starting  n={len(A_int):,}  m={len(B_int):,}", flush=True)

    t0 = time.perf_counter()
    best_score, path = run_dp(A_int, B_int, ALPHA, BETA, GAP_PENALTY, eps, hard_eps=False)
    runtime = time.perf_counter() - t0
    print(f"[V1] DP done in {runtime:.1f}s  score={best_score:.0f}", flush=True)

    # ---- collect stats ----
    n_match = n_gap_a = n_gap_b = 0
    errors  = []
    trace_rows = []

    for step, (op, i, j) in enumerate(path):
        if op == 'MATCH':
            n_match += 1
            ai  = int(A_int[i])
            bj  = int(B_int[j])
            err = abs(ai - bj)
            errors.append(err)
            trace_rows.append((step, 'MATCH', i, j, ai, bj, err))
        elif op == 'GAP_A':
            n_gap_a += 1
            trace_rows.append((step, 'GAP_A', None, j, None, int(B_int[j]), None))
        else:
            n_gap_b += 1
            trace_rows.append((step, 'GAP_B', i, None, int(A_int[i]), None, None))

    # ---- trace file ----
    trace_path = os.path.join(outdir, 'alignment_trace.tsv')
    with open(trace_path, 'w') as f:
        f.write('step\top\tA_int_idx\tB_int_idx\tA_interval\tB_interval\terror\n')
        for row in trace_rows:
            step, op, i, j, ai, bj, err = row
            f.write(f"{step}\t{op}\t{i or 'NA'}\t{j or 'NA'}\t"
                    f"{ai or 'NA'}\t{bj or 'NA'}\t{err if err is not None else 'NA'}\n")

    # ---- anchor coord map (for dotplot) ----
    anchors = make_anchor_pairs(path)
    anchor_path = os.path.join(outdir, 'anchor_coords.tsv')
    with open(anchor_path, 'w') as f:
        f.write('A_coord_idx\tB_coord_idx\tA_coord\tB_coord\n')
        for ia, jb in anchors:
            if 0 <= ia < len(A_coords) and 0 <= jb < len(B_coords):
                f.write(f"{ia}\t{jb}\t{int(A_coords[ia])}\t{int(B_coords[jb])}\n")
    print(f"[V1] anchor map written: {len(anchors):,} pairs", flush=True)

    # ---- gap position file (for clustering analysis) ----
    gap_path = os.path.join(outdir, 'gap_positions.tsv')
    with open(gap_path, 'w') as f:
        f.write('step\top\tA_coord\tB_coord\n')
        for row in trace_rows:
            step, op, i, j, ai, bj, err = row
            if op == 'GAP_B':   # gap in B sequence: A site has no match
                ac = int(A_coords[i])   if i is not None and i < len(A_coords) else 'NA'
                f.write(f"{step}\t{op}\t{ac}\tNA\n")
            elif op == 'GAP_A': # gap in A sequence: B site has no match
                bc = int(B_coords[j])   if j is not None and j < len(B_coords) else 'NA'
                f.write(f"{step}\t{op}\tNA\t{bc}\n")

    # ---- summary ----
    metrics = compute_proxy_metrics(
        len(A_coords), len(B_coords), n_match, n_gap_a, n_gap_b, errors, eps, runtime
    )
    metrics['best_score'] = float(best_score)
    summary_path = os.path.join(outdir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[V1] match_rate_A={metrics['match_rate_A']:.3f}  "
          f"gaps_A={n_gap_a:,}  gaps_B={n_gap_b:,}  "
          f"mean_err={metrics.get('error_mean_bp','N/A')}bp  "
          f"frac_within_eps={metrics.get('frac_within_eps','N/A')}", flush=True)
    print(f"[V1] outputs → {outdir}", flush=True)
    return metrics, anchors


# ---------------------------------------------------------------------------
# V2 runner
# ---------------------------------------------------------------------------

def run_v2(A_coords, B_coords, A_int, B_int, outdir, eps):
    os.makedirs(outdir, exist_ok=True)
    print(f"[V2] starting  n={len(A_int):,}  m={len(B_int):,}", flush=True)

    t0 = time.perf_counter()
    best_score, path = run_dp_v2(
        A_int, B_int, ALPHA, SIGMA, GAMMA, LAMBDA_A, LAMBDA_B, eps, GAP_PENALTY
    )
    runtime = time.perf_counter() - t0
    print(f"[V2] DP done in {runtime:.1f}s  score={best_score:.0f}", flush=True)

    n_match = n_merge = n_gap_a = n_gap_b = 0
    errors  = []
    anchors = set()

    trace_path = os.path.join(outdir, 'alignment_trace.tsv')
    with open(trace_path, 'w') as f:
        f.write('step\ttype\tA_idx_range\tB_idx_range\tdistA\tdistB\terror\n')
        for step, move in enumerate(path):
            pi, pj = move['prev']
            ci, cj = move['curr']
            mtype  = move['type']
            dA = int(np.sum(A_int[pi:ci]))
            dB = int(np.sum(B_int[pj:cj]))
            err = abs(dA - dB)
            f.write(f"{step}\t{mtype}\t{pi}:{ci}\t{pj}:{cj}\t{dA}\t{dB}\t{err}\n")

            if mtype == 'MATCH':
                n_match += 1;  errors.append(err)
                for ai, bj in [(pi, pj), (ci, cj)]:
                    if 0 <= ai < len(A_coords) and 0 <= bj < len(B_coords):
                        anchors.add((ai, bj))
            elif mtype == 'MERGE':
                n_merge += 1;  errors.append(err)
                for ai, bj in [(pi, pj), (ci, cj)]:
                    if 0 <= ai < len(A_coords) and 0 <= bj < len(B_coords):
                        anchors.add((ai, bj))
            elif mtype == 'GAP_A':
                n_gap_a += 1
            elif mtype == 'GAP_B':
                n_gap_b += 1

    # ---- anchor coord map (for dotplot) ----
    anchor_path = os.path.join(outdir, 'anchor_coords.tsv')
    with open(anchor_path, 'w') as f:
        f.write('A_coord_idx\tB_coord_idx\tA_coord\tB_coord\n')
        for ia, jb in sorted(anchors):
            f.write(f"{ia}\t{jb}\t{int(A_coords[ia])}\t{int(B_coords[jb])}\n")
    print(f"[V2] anchor map written: {len(anchors):,} pairs", flush=True)

    metrics = compute_proxy_metrics(
        len(A_coords), len(B_coords), n_match + n_merge,
        n_gap_a, n_gap_b, errors, eps, runtime
    )
    metrics['best_score'] = float(best_score)
    metrics['n_merge']    = int(n_merge)
    summary_path = os.path.join(outdir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[V2] match_rate_A={metrics['match_rate_A']:.3f}  "
          f"merges={n_merge:,}  gaps_A={n_gap_a:,}  gaps_B={n_gap_b:,}  "
          f"mean_err={metrics.get('error_mean_bp','N/A')}bp  "
          f"frac_within_eps={metrics.get('frac_within_eps','N/A')}", flush=True)
    print(f"[V2] outputs → {outdir}", flush=True)
    return metrics, anchors


# ---------------------------------------------------------------------------
# Agreement analysis (only when both algorithms are run)
# ---------------------------------------------------------------------------

def compute_agreement(anchors_v1, anchors_v2, outdir):
    a1 = frozenset(anchors_v1)
    a2 = frozenset(anchors_v2)
    agree   = a1 & a2
    only_v1 = a1 - a2
    only_v2 = a2 - a1

    total   = len(a1 | a2)
    result  = {
        'n_agree'           : len(agree),
        'n_only_v1'         : len(only_v1),
        'n_only_v2'         : len(only_v2),
        'agreement_rate'    : round(len(agree) / total, 4) if total else 0,
    }
    path = os.path.join(outdir, 'agreement.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[Agreement] {len(agree):,} shared pairs "
          f"({result['agreement_rate']:.1%} of union)  "
          f"V1-only={len(only_v1):,}  V2-only={len(only_v2):,}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run alignment on real E. coli data')
    parser.add_argument('--alg',     choices=['v1', 'v2', 'both'], default='v1')
    parser.add_argument('--eps',     type=int,   default=EPS)
    parser.add_argument('--data-dir', type=str,  default=DATA_DIR)
    parser.add_argument('--out-dir',  type=str,  default=OUT_ROOT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("==========================================", flush=True)
    print(f"Job ID     : {os.environ.get('SLURM_JOB_ID', 'local')}", flush=True)
    print(f"Algorithm  : {args.alg}", flush=True)
    print(f"Data dir   : {args.data_dir}", flush=True)
    print(f"Out dir    : {args.out_dir}", flush=True)
    print(f"eps        : {args.eps}", flush=True)
    print("==========================================", flush=True)

    A_coords = load_1d(os.path.join(args.data_dir, 'A_coords.csv'))
    B_coords = load_1d(os.path.join(args.data_dir, 'B_coords.csv'))
    A_int    = intervals_from_coords(A_coords)
    B_int    = intervals_from_coords(B_coords)

    print(f"\nA: {len(A_coords):,} sites  ({len(A_int):,} intervals)", flush=True)
    print(f"B: {len(B_coords):,} sites  ({len(B_int):,} intervals)\n", flush=True)

    anchors_v1 = anchors_v2 = None

    if args.alg in ('v1', 'both'):
        v1_outdir = os.path.join(args.out_dir, 'v1')
        _, anchors_v1 = run_v1(A_coords, B_coords, A_int, B_int, v1_outdir, args.eps)

    if args.alg in ('v2', 'both'):
        v2_outdir = os.path.join(args.out_dir, 'v2')
        _, anchors_v2 = run_v2(A_coords, B_coords, A_int, B_int, v2_outdir, args.eps)

    if args.alg == 'both' and anchors_v1 and anchors_v2:
        compute_agreement(anchors_v1, anchors_v2, args.out_dir)

    print("\nAll done.", flush=True)


if __name__ == '__main__':
    main()
