#!/bin/bash
#SBATCH --job-name=epi_benchmark
#SBATCH --chdir=/new-stg/home/hanbei/class_project/epigenome_alignment
#SBATCH --output=/new-stg/home/hanbei/class_project/epigenome_alignment/logs/benchmark_%j.out
#SBATCH --error=/new-stg/home/hanbei/class_project/epigenome_alignment/logs/benchmark_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

source /new-stg/home/hanbei/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : $(hostname)"
echo "Start time : $(date)"
echo "Working dir: $(pwd)"
echo "Python     : $(which python)"
echo "=========================================="

python -u src/benchmark.py

echo ""
echo "Finished at: $(date)"
