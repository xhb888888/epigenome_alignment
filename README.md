# Epigenome Alignment

Pairwise alignment of ordered genomic coordinate lists (epigenetic motif sites) using dynamic programming. Implements and benchmarks two algorithms — a **naive global DP (V1)** and a **banded merge DP (V2)** — and evaluates them on both synthetic datasets and real *E. coli* GATC methylation data.

---

## Background

Epigenetic motif sites (e.g., GATC Dam-methylation sites) can be represented as ordered lists of genomic coordinates. Aligning two such lists reveals syntenic correspondence between genomes: which site in genome A maps to which site in genome B. Unlike sequence alignment, the "characters" being aligned are inter-site interval lengths, and the challenge is to handle inserted/deleted sites (motif gain/loss) and large structural rearrangements.

---

## Algorithms

### V1 — Naive Global DP (`src/naive_dp.py`)

A classic global alignment (Needleman–Wunsch style) over the sequence of **inter-site intervals**.

- **Match score:** `alpha - beta * |interval_A - interval_B|`
- **Gap penalty:** constant `gap_penalty` per skipped site
- **Complexity:** O(n × m) time and space

Default parameters: `alpha=100`, `beta=1.0`, `gap_penalty=80`, `eps=50 bp`

### V2 — Banded Merge DP (`src/run_dp_v2.py`)

Extends V1 with **merge moves**: a single alignment step can span multiple consecutive intervals on one or both sides. This natively handles motif gain/loss (split or merged intervals) without incurring gap penalties for every extra site.

- **Merge penalty:** `gamma + lambda_a * (di - 1) + lambda_b * (dj - 1)` for a di:dj merge
- **Band restriction:** only cells within `|n - m| + 200` of the diagonal are filled
- **Fallback gaps:** explicit GAP_A / GAP_B moves for sites that cannot be merged
- **Move types in backtrack:** `MATCH`, `MERGE`, `GAP_A`, `GAP_B`

Default parameters: `alpha=100`, `sigma=1.0`, `gamma=10`, `lambda_a=2`, `lambda_b=2`, `gap_penalty=80`, `eps=50 bp`, `window=15`

---

## Repository Structure

```
epigenome_alignment/
├── src/
│   ├── naive_dp.py         # V1: naive global DP alignment
│   ├── run_dp_v2.py        # V2: banded merge DP alignment
│   ├── benchmark.py        # Synthetic benchmark runner
│   └── run_real.py         # Real E. coli data runner
│
├── data/
│   ├── synthetic/
│   │   ├── clean/          # B = A exactly (sanity check)
│   │   ├── missing_sites/  # 10% of B sites deleted (motif loss)
│   │   ├── extra_sites/    # 5% extra sites inserted into B (motif gain)
│   │   └── big_indel/      # 50,000 bp suffix shift in B (structural rearrangement)
│   └── real/
│       └── ecoli_realdata_synth_format/   # E. coli K-12 vs O157:H7 GATC sites
│
├── results/
│   └── real/
│       ├── v1/             # V1 output: trace, anchors, gap positions, summary
│       └── v2/             # V2 output: trace, anchors, summary
│
├── notebook/
│   ├── benchmark_results.ipynb     # Synthetic benchmark visualisations
│   └── real_data_results.ipynb     # Real data visualisations
│
├── logs/                   # SLURM job logs
├── run_benchmark.sh        # SLURM script: synthetic benchmark
├── run_real_v1.sh          # SLURM script: V1 on real data
├── run_real_v2.sh          # SLURM script: V2 on real data
└── benchmark_results.json  # Saved benchmark results
```

---

## Synthetic Datasets

Four controlled scenarios (2,000 A-sites, mean interval 250 bp, 0-based coordinates):

| Scenario | Description | A sites | B sites |
|---|---|---|---|
| `clean` | B = A exactly — sanity check | 2,000 | 2,000 |
| `missing_sites` | 10% of internal B sites deleted — simulates motif loss | 2,000 | 1,794 |
| `extra_sites` | New sites inserted into B at 5% rate — simulates motif gain | 2,000 | 2,104 |
| `big_indel` | B suffix shifted by 50,000 bp — simulates a large structural rearrangement | 2,000 | 2,000 |

Ground truth is inferred from the data-generation model: exact-coordinate matches for `missing_sites`/`extra_sites`, index-identity for `clean`/`big_indel`.

---

## Real Dataset

**E. coli K-12 MG1655 vs O157:H7 EDL933 — GATC (Dam methylation) sites**

| Genome | Sites |
|---|---|
| K-12 MG1655 (A) | 19,124 |
| O157:H7 EDL933 (B) | 21,372 |

No ground truth is available; performance is evaluated via proxy metrics (match rate, error distribution, gap clustering, algorithm agreement).

---

## Benchmark Results (Synthetic)

| Dataset | Algorithm | F1 | Precision | Recall | Mean Error (bp) | Runtime (s) |
|---|---|---|---|---|---|---|
| clean | V1 | 1.000 | 1.000 | 1.000 | 0.0 | 5.0 |
| clean | V2 | 1.000 | 1.000 | 1.000 | 0.0 | 149.1 |
| missing_sites | V1 | 0.953 | 0.917 | 0.992 | 0.0 | 4.5 |
| missing_sites | **V2** | **1.000** | **1.000** | **1.000** | 0.0 | 247.7 |
| extra_sites | V1 | 0.975 | 0.952 | 0.999 | 0.0 | 5.1 |
| extra_sites | **V2** | **1.000** | **1.000** | **1.000** | 0.0 | 230.2 |
| big_indel | V1 | 1.000 | 1.000 | 1.000 | 25,000.0 | 4.8 |
| big_indel | V2 | 1.000 | 1.000 | 1.000 | 25,000.0 | 147.6 |

**Key findings:**
- V2 achieves perfect F1 on `missing_sites` and `extra_sites` where V1 produces false positives (F1 ≈ 0.95–0.97) due to its inability to natively handle interval splitting/merging.
- Both algorithms handle `big_indel` equivalently (large mean error reflects the 50,000 bp structural shift by design, not misalignment).
- V1 is **30–50× faster** than V2 at this scale.

---

## Real Data Results (E. coli)

| Metric | V1 | V2 |
|---|---|---|
| Match Rate (A sites) | 90.9% | 87.1% |
| Frac within eps=50 bp | 89.8% | **100.0%** |
| Exact Match Fraction | 75.4% | **83.5%** |
| Mean Interval Error (bp) | 14.0 | **3.0** |
| Median Interval Error (bp) | 0.0 | 0.0 |
| P90 Error (bp) | 52 | **11** |
| P99 Error (bp) | 186 | **44** |
| Runtime | 9.2 min | 2.4 hr |
| Matched pairs | 17,374 | 16,651 |
| Gaps in A | 3,997 | 1,743 |
| Gaps in B | 1,749 | 392 |
| Merge moves (V2 only) | — | 2,563 |

**Key findings:**
- V2 produces dramatically tighter error distributions: P90 error drops from 52 bp to 11 bp, P99 from 186 bp to 44 bp, and all matches fall within 50 bp (frac_within_eps = 1.0).
- V2 uses 2,563 merge moves, consolidating split/merged interval groups that V1 resolves with gaps — explaining why V1 accumulates more gaps and higher error.
- The trade-off is runtime: V2 takes ~15× longer (2.4 hr vs 9.2 min) at the full 19K × 21K scale.

---

## Usage

### Prerequisites

```bash
conda activate base   # any environment with numpy, pandas, scipy
```

### Run alignment on custom data

**V1:**
```bash
python src/naive_dp.py \
    --a-coords data/real/ecoli_realdata_synth_format/A_coords.csv \
    --b-coords data/real/ecoli_realdata_synth_format/B_coords.csv \
    --outdir results/my_run/v1
```

**V2:**
```bash
python src/run_dp_v2.py \
    --a-coords data/real/ecoli_realdata_synth_format/A_coords.csv \
    --b-coords data/real/ecoli_realdata_synth_format/B_coords.csv \
    --outdir results/my_run/v2
```

Key options shared by both: `--eps` (match tolerance in bp, default 50), `--gap-penalty`, `--alpha`.
V2 additionally supports: `--gamma`, `--lambda-a`, `--lambda-b`, `--sigma`.

### Run the synthetic benchmark

```bash
# Locally
python src/benchmark.py
```

Results are written to `benchmark_results.json`.

### Run on real E. coli data

```bash
# Locally
python src/run_real.py --alg v1   # V1 only
python src/run_real.py --alg v2   # V2 only
python src/run_real.py --alg both # both + agreement stats
```

Outputs go to `results/real/v1/` and `results/real/v2/`:
- `alignment_trace.tsv` — per-step move log (MATCH / MERGE / GAP_A / GAP_B)
- `anchor_coords.tsv` — matched (A_coord, B_coord) pairs
- `gap_positions.tsv` — genomic positions of gap events (V1 only)
- `summary.json` — aggregate metrics

### Visualise results

Open the notebooks in Jupyter:

```bash
jupyter notebook notebook/benchmark_results.ipynb    # synthetic results
jupyter notebook notebook/real_data_results.ipynb    # real data results
```

---

## Output Files

| File | Description |
|---|---|
| `alignment_trace.tsv` | Full backtrack path: one row per DP move with move type, interval indices, and coordinate values |
| `anchor_coords.tsv` | MATCH-derived anchor pairs: `A_coord_idx`, `B_coord_idx`, `A_coord`, `B_coord` |
| `gap_positions.tsv` | Gap events with genomic position and type (GAP_A / GAP_B) |
| `summary.json` | Aggregate metrics: match counts, error statistics, runtime, DP score |
| `benchmark_results.json` | Full synthetic benchmark table (all datasets × both algorithms) |
