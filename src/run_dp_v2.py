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

def run_dp_v2(A_int, B_int, alpha, sigma, gamma, lambda_a, lambda_b, eps):
    n, m = len(A_int), len(B_int)
    NEG = -1e15
    dp = np.full((n + 1, m + 1), NEG, dtype=float)
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    A_prefix = np.concatenate(([0], np.cumsum(A_int)))
    window = 15

    for i in range(1, n + 1):
        pi_start = max(0, i - window)

        band = abs(n - m) + 200
        j_start = max(1, i - band)
        j_end = min(m + 1, i + band)

        for j in range(j_start, j_end):

            pj_start = max(0, j - window)

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

    path = []
    curr = (n, m)
    while curr != (0, 0):
        prev = bt[curr[0]][curr[1]]
        if prev is None: break
        path.append({'prev': prev, 'curr': curr})
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
    best_score, path = run_dp_v2(A_int, B_int, args.alpha, args.sigma, args.gamma, args.lambda_a, args.lambda_b, args.eps)
    end_time = time.perf_counter()
    runtime = end_time - start_time
    print(f"Time taken: {runtime:.2f} seconds")

    # Statistics and File Writing
    # Statistics and File Writing
    trace_file = os.path.join(args.outdir, 'alignment_trace.tsv')
    pairs_file = os.path.join(args.outdir, 'alignment_pairs.tsv')

    match_rows = []
    n_match = 0
    n_merge = 0

    with open(trace_file, 'w') as f:
        f.write("step\ttype\tA_idx_range\tB_idx_range\tdistA\tdistB\terror\n")

        for step, move in enumerate(path):
            pi, pj = move['prev']
            ci, cj = move['curr']

            dA = np.sum(A_int[pi:ci])
            dB = np.sum(B_int[pj:cj])
            err = abs(dA - dB)

            if (ci - pi == 1 and cj - pj == 1):
                m_type = "MATCH"
                n_match += 1
            else:
                m_type = "MERGE"
                n_merge += 1

            f.write(f"{step}\t{m_type}\t{pi}:{ci}\t{pj}:{cj}\t{dA}\t{dB}\t{err}\n")

            match_rows.append((step, pi, pj, dA, dB, err))

    with open(pairs_file, 'w') as f:
        f.write("step\tA_idx\tB_idx\tA_dist\tB_dist\terror\twithin_eps\n")

        for row in match_rows:
            step, pi, pj, dA, dB, err = row
            within_eps = 1 if err <= args.eps else 0

            f.write(
                f"{step}\t{pi}\t{pj}\t{dA}\t{dB}\t{err}\t{within_eps}\n"
            )
    mean_abs_error = None
    median_abs_error = None
    frac_within_eps = None
    if match_rows:
        errs = [r[5] for r in match_rows]

        mean_abs_error = float(np.mean(errs))
        median_abs_error = float(np.median(errs))
        frac_within_eps = float(np.mean(np.array(errs) <= args.eps))

        print(f"Mean abs error: {mean_abs_error:.2f} bp")
        print(f"Median abs error: {median_abs_error:.2f} bp")
        print(f"Fraction within eps={args.eps}: {frac_within_eps:.3f}")
    print(f"Best score: {best_score:.2f}")
    print(f"Matches: {n_match}")
    print(f"Merges: {n_merge}")

    summary = {
        "best_score": float(best_score),
        "runtime_seconds": runtime,
        "n_steps": len(path),
        "n_match": int(n_match),
        "n_merge": int(n_merge),
        "mean_abs_error": mean_abs_error,
        "median_abs_error": median_abs_error,
        "frac_within_eps": frac_within_eps
    }

    summary_file = os.path.join(args.outdir, "alignment_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == '__main__':
    main()