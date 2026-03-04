#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
import time
def load_1d(path):
    if path.endswith('.npy'):
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=',')
        except Exception:
            arr = np.loadtxt(path)
    return np.array(arr).reshape(-1).astype(np.int64)

def check_coords(coords, name):
    if len(coords) < 2:
        raise ValueError(f"{name} needs at least 2 values")
    if not np.all(np.diff(coords) > 0):
        raise ValueError(f"{name} is not strictly increasing")

def intervals_from_coords(coords):
    return np.diff(coords).astype(np.int64)

def run_dp_v2(A_int, B_int, alpha, sigma, gamma, lambda_a, lambda_b, eps,
              gap_penalty=80.0):
    n, m = len(A_int), len(B_int)
    NEG = -1e15
    dp = np.full((n + 1, m + 1), NEG, dtype=float)
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    A_prefix = np.concatenate(([0], np.cumsum(A_int)))
    window = 15

    # Initialize boundaries with gap moves so paths can start with gaps
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_penalty
        bt[i][0] = (i - 1, 0)           # GAP_B: dj == 0
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_penalty
        bt[0][j] = (0, j - 1)           # GAP_A: di == 0

    for i in range(1, n + 1):
        pi_start = max(0, i - window)

        band = abs(n - m) + 200
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band)

        for j in range(j_start, j_end):

            pj_start = max(0, j - window)

            # --- merge / match moves (existing logic) ---
            for pi in range(pi_start, i):

                distA = A_prefix[i] - A_prefix[pi]

                distB = 0
                for pj in range(j-1, pj_start-1, -1):

                    distB += B_int[pj]

                    if distB - distA > eps:
                        break

                    if dp[pi][pj] == NEG:
                        continue

                    err = abs(distA - distB)
                    if err > eps:
                        continue

                    di = i - pi
                    dj = j - pj

                    p_t = 0 if (di == 1 and dj == 1) else (
                        gamma + lambda_a*(di-1) + lambda_b*(dj-1)
                    )

                    score = dp[pi][pj] + (alpha - sigma*err) - p_t

                    if score > dp[i][j]:
                        dp[i][j] = score
                        bt[i][j] = (pi, pj)

            # --- gap fallback moves ---
            # GAP_B: skip A_int[i-1], come from dp[i-1][j]
            if dp[i - 1][j] > NEG:
                gap_score = dp[i - 1][j] - gap_penalty
                if gap_score > dp[i][j]:
                    dp[i][j] = gap_score
                    bt[i][j] = (i - 1, j)   # dj == 0  →  GAP_B

            # GAP_A: skip B_int[j-1], come from dp[i][j-1]
            # dp[i][j-1] is valid when j-1 >= j_start (already filled this row)
            if dp[i][j - 1] > NEG:
                gap_score = dp[i][j - 1] - gap_penalty
                if gap_score > dp[i][j]:
                    dp[i][j] = gap_score
                    bt[i][j] = (i, j - 1)   # di == 0  →  GAP_A

    # Backtrack
    path = []
    curr = (n, m)
    while curr != (0, 0):
        prev = bt[curr[0]][curr[1]]
        if prev is None:
            break
        pi, pj = prev
        ci, cj = curr
        di, dj = ci - pi, cj - pj
        if di == 0:
            move_type = 'GAP_A'
        elif dj == 0:
            move_type = 'GAP_B'
        elif di == 1 and dj == 1:
            move_type = 'MATCH'
        else:
            move_type = 'MERGE'
        path.append({'prev': prev, 'curr': curr, 'type': move_type})
        curr = prev
    path.reverse()
    return dp[n][m], path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a-coords', type=str)
    parser.add_argument('--b-coords', type=str)
    parser.add_argument('--a-intervals', type=str)
    parser.add_argument('--b-intervals', type=str)
    parser.add_argument('--eps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=100.0)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=10.0)
    parser.add_argument('--lambda-a', type=float, default=2.0)
    parser.add_argument('--lambda-b', type=float, default=2.0)
    parser.add_argument('--gap-penalty', type=float, default=80.0)
    parser.add_argument('--outdir', type=str, default='v2_dp_out')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)

    A_coords, B_coords = None, None
    if args.a_intervals and args.b_intervals:
        A_int, B_int = load_1d(args.a_intervals), load_1d(args.b_intervals)
    elif args.a_coords and args.b_coords:
        A_coords, B_coords = load_1d(args.a_coords), load_1d(args.b_coords)
        check_coords(A_coords, 'A_coords'); check_coords(B_coords, 'B_coords')
        A_int, B_int = intervals_from_coords(A_coords), intervals_from_coords(B_coords)
    else:
        raise ValueError('Need coords or intervals')
    start_time = time.perf_counter()
    best_score, path = run_dp_v2(A_int, B_int, args.alpha, args.sigma, args.gamma, args.lambda_a, args.lambda_b, args.eps, args.gap_penalty)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Time taken: {runtime:.2f} seconds")

    # Statistics and File Writing
    # Statistics and File Writing
    trace_file = os.path.join(args.outdir, 'alignment_trace.tsv')
    pairs_file = os.path.join(args.outdir, 'alignment_pairs.tsv')

    align_rows = []   # MATCH and MERGE only (for pairs_file and error stats)
    n_match  = 0
    n_merge  = 0
    n_gap_a  = 0
    n_gap_b  = 0

    with open(trace_file, 'w') as f:
        f.write("step\ttype\tA_idx_range\tB_idx_range\tdistA\tdistB\terror\n")

        for step, move in enumerate(path):
            pi, pj = move['prev']
            ci, cj = move['curr']
            m_type = move['type']

            dA = int(np.sum(A_int[pi:ci]))
            dB = int(np.sum(B_int[pj:cj]))
            err = abs(dA - dB)

            if m_type == 'MATCH':
                n_match += 1
            elif m_type == 'MERGE':
                n_merge += 1
            elif m_type == 'GAP_A':
                n_gap_a += 1
            elif m_type == 'GAP_B':
                n_gap_b += 1

            f.write(f"{step}\t{m_type}\t{pi}:{ci}\t{pj}:{cj}\t{dA}\t{dB}\t{err}\n")

            if m_type in ('MATCH', 'MERGE'):
                align_rows.append((step, pi, pj, dA, dB, err))

    with open(pairs_file, 'w') as f:
        f.write("step\tA_idx\tB_idx\tA_dist\tB_dist\terror\twithin_eps\n")
        for row in align_rows:
            step, pi, pj, dA, dB, err = row
            within_eps = 1 if err <= args.eps else 0
            f.write(f"{step}\t{pi}\t{pj}\t{dA}\t{dB}\t{err}\t{within_eps}\n")

    mean_abs_error = None
    median_abs_error = None
    frac_within_eps = None
    if align_rows:
        errs = [r[5] for r in align_rows]
        mean_abs_error   = float(np.mean(errs))
        median_abs_error = float(np.median(errs))
        frac_within_eps  = float(np.mean(np.array(errs) <= args.eps))
        print(f"Mean abs error: {mean_abs_error:.2f} bp")
        print(f"Median abs error: {median_abs_error:.2f} bp")
        print(f"Fraction within eps={args.eps}: {frac_within_eps:.3f}")
    print(f"Best score: {best_score:.2f}")
    print(f"Matches: {n_match}  Merges: {n_merge}  Gap_A: {n_gap_a}  Gap_B: {n_gap_b}")

    summary = {
        "best_score": float(best_score),
        "runtime_seconds": runtime,
        "n_steps": len(path),
        "n_match": int(n_match),
        "n_merge": int(n_merge),
        "n_gap_a": int(n_gap_a),
        "n_gap_b": int(n_gap_b),
        "mean_abs_error": mean_abs_error,
        "median_abs_error": median_abs_error,
        "frac_within_eps": frac_within_eps
    }

    summary_file = os.path.join(args.outdir, "alignment_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()