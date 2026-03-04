#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np


def load_1d(path):
    if path.endswith('.npy'):
        arr = np.load(path)
    else:
        try:
            arr = np.loadtxt(path, delimiter=',')
        except Exception:
            arr = np.loadtxt(path)

    arr = np.array(arr).reshape(-1)
    return arr.astype(np.int64)


def check_coords(coords, name):
    if len(coords) < 2:
        raise ValueError(name + ' needs at least 2 values')

    for i in range(1, len(coords)):
        if coords[i] <= coords[i - 1]:
            raise ValueError(name + ' is not strictly increasing')


def intervals_from_coords(coords):
    out = []
    for i in range(len(coords) - 1):
        out.append(int(coords[i + 1] - coords[i]))
    return np.array(out, dtype=np.int64)


def score(a, b, alpha, beta):
    err = abs(int(a) - int(b))
    return alpha - beta * err


def run_dp(A_int, B_int, alpha, beta, gap_penalty, eps, hard_eps):
    n = len(A_int)
    m = len(B_int)

    NEG = -10**15

    dp = []
    bt = []
    for _ in range(n + 1):
        dp.append([NEG] * (m + 1))
        bt.append([None] * (m + 1))

    dp[0][0] = 0.0

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - gap_penalty
        bt[i][0] = 'UP'

    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - gap_penalty
        bt[0][j] = 'LEFT'

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            this_score = score(A_int[i - 1], B_int[j - 1], alpha, beta)
            if hard_eps:
                if abs(int(A_int[i - 1]) - int(B_int[j - 1])) > eps:
                    this_score = NEG

            diag = dp[i - 1][j - 1] + this_score
            up = dp[i - 1][j] - gap_penalty
            left = dp[i][j - 1] - gap_penalty

            best = diag
            move = 'DIAG'

            if up > best:
                best = up
                move = 'UP'
            if left > best:
                best = left
                move = 'LEFT'

            dp[i][j] = best
            bt[i][j] = move

    # backtrack from homework
    i = n
    j = m
    path = []
    while i > 0 or j > 0:
        move = bt[i][j]

        if move == 'DIAG':
            path.append(('MATCH', i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == 'UP':
            path.append(('GAP_B', i - 1, None))
            i -= 1
        elif move == 'LEFT':
            path.append(('GAP_A', None, j - 1))
            j -= 1
        else:
            if i > 0:
                path.append(('GAP_B', i - 1, None))
                i -= 1
            else:
                path.append(('GAP_A', None, j - 1))
                j -= 1

    path.reverse()
    return dp[n][m], path


def make_anchor_pairs(path):
    pairs = []
    for op, i, j in path:
        if op == 'MATCH':
            pairs.append((i, j))
            pairs.append((i + 1, j + 1))

    unique_pairs = []
    seen = set()
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique_pairs.append(p)

    return unique_pairs


def main():
    parser = argparse.ArgumentParser()
    # command line usages 
    parser.add_argument('--a-coords', type=str)
    parser.add_argument('--b-coords', type=str)
    parser.add_argument('--a-intervals', type=str)
    parser.add_argument('--b-intervals', type=str)
    parser.add_argument('--meta-json', type=str)

    parser.add_argument('--eps', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=100.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gap-penalty', type=float, default=80.0)
    parser.add_argument('--hard-eps', action='store_true')
    parser.add_argument('--outdir', type=str, default='naive_dp_out')

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    A_coords = None
    B_coords = None

    if args.a_intervals and args.b_intervals:
        A_int = load_1d(args.a_intervals)
        B_int = load_1d(args.b_intervals)
    elif args.a_coords and args.b_coords:
        A_coords = load_1d(args.a_coords)
        B_coords = load_1d(args.b_coords)

        check_coords(A_coords, 'A_coords')
        check_coords(B_coords, 'B_coords')

        A_int = intervals_from_coords(A_coords)
        B_int = intervals_from_coords(B_coords)
    else:
        raise ValueError('Need either coords pair or intervals pair')

    if len(A_int) == 0 or len(B_int) == 0:
        raise ValueError('Empty interval array')

    best_score, path = run_dp(
        A_int,
        B_int,
        args.alpha,
        args.beta,
        args.gap_penalty,
        args.eps,
        args.hard_eps,
    )

    # collect stats
    n_match = 0
    n_gap_a = 0
    n_gap_b = 0
    match_rows = []

    for step in range(len(path)):
        op, i, j = path[step]
        if op == 'MATCH':
            n_match += 1
            ai = int(A_int[i])
            bj = int(B_int[j])
            err = abs(ai - bj)
            s = score(ai, bj, args.alpha, args.beta)
            if args.hard_eps and err > args.eps:
                s = -10**15
            match_rows.append((step, i, j, ai, bj, err, s))
        elif op == 'GAP_A':
            n_gap_a += 1
        else:
            n_gap_b += 1

    trace_file = os.path.join(args.outdir, 'alignment_trace.tsv')
    with open(trace_file, 'w') as f:
        f.write('step\top\tA_interval_idx\tB_interval_idx\tA_interval\tB_interval\terror\tscore\n')
        for step in range(len(path)):
            op, i, j = path[step]
            if op == 'MATCH':
                ai = int(A_int[i])
                bj = int(B_int[j])
                err = abs(ai - bj)
                s = score(ai, bj, args.alpha, args.beta)
                if args.hard_eps and err > args.eps:
                    s = -10**15
                f.write(str(step) + '\tMATCH\t' + str(i) + '\t' + str(j) + '\t' + str(ai) + '\t' + str(bj) + '\t' + str(err) + '\t' + format(s, '.3f') + '\n')
            elif op == 'GAP_B':
                ai = int(A_int[i])
                f.write(str(step) + '\tGAP_B\t' + str(i) + '\tNA\t' + str(ai) + '\tNA\tNA\t' + format(-args.gap_penalty, '.3f') + '\n')
            else:
                bj = int(B_int[j])
                f.write(str(step) + '\tGAP_A\tNA\t' + str(j) + '\tNA\t' + str(bj) + '\tNA\t' + format(-args.gap_penalty, '.3f') + '\n')

    pairs_file = os.path.join(args.outdir, 'alignment_pairs.tsv')
    with open(pairs_file, 'w') as f:
        f.write('step\tA_interval_idx\tB_interval_idx\tA_interval\tB_interval\terror\twithin_eps\tscore\n')
        for row in match_rows:
            step, i, j, ai, bj, err, s = row
            within_eps = 1 if err <= args.eps else 0
            f.write(str(step) + '\t' + str(i) + '\t' + str(j) + '\t' + str(ai) + '\t' + str(bj) + '\t' + str(err) + '\t' + str(within_eps) + '\t' + format(s, '.3f') + '\n')

    anchor_file = None
    if A_coords is not None and B_coords is not None:
        anchor_file = os.path.join(args.outdir, 'anchor_mapping_coords.tsv')
        anchors = make_anchor_pairs(path)
        with open(anchor_file, 'w') as f:
            f.write('A_coord_idx\tB_coord_idx\tA_coord\tB_coord\n')
            for ia, jb in anchors:
                if ia >= 0 and jb >= 0 and ia < len(A_coords) and jb < len(B_coords):
                    f.write(str(ia) + '\t' + str(jb) + '\t' + str(int(A_coords[ia])) + '\t' + str(int(B_coords[jb])) + '\n')

    summary = {
        'best_score': float(best_score),
        'A_num_intervals': int(len(A_int)),
        'B_num_intervals': int(len(B_int)),
        'n_match': int(n_match),
        'n_gap_a': int(n_gap_a),
        'n_gap_b': int(n_gap_b),
        'eps': int(args.eps),
        'alpha': float(args.alpha),
        'beta': float(args.beta),
        'gap_penalty': float(args.gap_penalty),
        'hard_eps': bool(args.hard_eps),
        'outputs': {
            'trace': trace_file,
            'pairs': pairs_file,
            'anchors': anchor_file,
        },
    }

    if len(match_rows) > 0:
        errs = [r[5] for r in match_rows]
        summary['mean_abs_error'] = float(np.mean(errs))
        summary['median_abs_error'] = float(np.median(errs))
        summary['frac_within_eps'] = float(np.mean(np.array(errs) <= args.eps))

    if args.meta_json and os.path.exists(args.meta_json):
        try:
            with open(args.meta_json, 'r') as f:
                summary['meta'] = json.load(f)
        except Exception as e:
            summary['meta_error'] = str(e)

    summary_file = os.path.join(args.outdir, 'alignment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Done')
    print('A intervals:', len(A_int))
    print('B intervals:', len(B_int))
    print('Best score:', round(best_score, 3))
    print('Matches:', n_match, 'gaps in A:', n_gap_a, 'gaps in B:', n_gap_b)
    if len(match_rows) > 0:
        errs = [r[5] for r in match_rows]
        print('Mean abs error:', round(float(np.mean(errs)), 2), 'bp')
        print('Fraction within eps=' + str(args.eps) + ':', round(float(np.mean(np.array(errs) <= args.eps)), 3))
    print('Saved to:', args.outdir)


if __name__ == '__main__':
    main()
