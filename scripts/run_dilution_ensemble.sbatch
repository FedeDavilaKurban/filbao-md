#!/bin/bash
# =====================================================================================================
# FilBAO - dilution ensemble on Clemente (IATE): random dilutions of Full, monopoles only.
#
# A SLURM job array. Task i computes ensemble members FIRST..LAST (CHUNK members) for every dilution
# size of the run in scripts/multidark_analysis.py (3 sizes in the cumulative run). Member j has seed
# dilution_seed + n_dilution_realisations + j - 1 (20261016 + j now) and writes
# ..._FullrandomN-<N>r<seed>.npz, exactly as n_dilution_ensemble does on the laptop.
# Members that already have their pair-count file are only reloaded (seconds), so:
#   - your existing members 1..50 cost nothing (tasks 0-4 finish at once);
#   - a task that crashes or times out can be resubmitted: sbatch --array=<ids> ...
# Ensemble = the original dilution + N_MEMBERS seeds = 1 + 499 = 500.
#
# Needs the FILBAO_ENS_* hook in scripts/multidark_analysis.py (ensemble-only mode).
#
# Use, on clemente, from the codes/ folder (the one with scripts/, src/, data/):
#   mkdir -p logs                                                             # SLURM does not create it
#   sbatch --partition=debug --time=0:02:00 --array=5 scripts/run_dilution_ensemble.sbatch  # start-up check
#   sbatch --array=5 scripts/run_dilution_ensemble.sbatch                     # one real task: read its time
#   sbatch scripts/run_dilution_ensemble.sbatch                               # everything (task 5 is reused)
#   squeue -u $USER ;  tail logs/dil_<jobid>_<task>.out
# =====================================================================================================
#SBATCH --job-name=filbao-dil
#SBATCH --partition=batch            # 3-day limit, clemente01-06
#SBATCH --time=08:00:00              # per task; laptop timing is ~1.5 h for 10 members x 3 sizes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # = the 16 threads Corrfunc takes on a 56-thread node (3 tasks per node)
#SBATCH --mem=16G
#SBATCH --array=0-49%12              # 50 tasks, at most 12 at once (leaves the cluster usable for others)
#SBATCH --output=logs/dil_%A_%a.out
#SBATCH --error=logs/dil_%A_%a.err

# ---------------- settings ---------------------------------------------------------------------------
N_MEMBERS=499                         # members j = 1..N_MEMBERS (plus the original dilution = 500)
CHUNK=10                              # members per task; --array must cover 0..ceil(N_MEMBERS/CHUNK)-1
CONDA_ENV=filbao2                     # env on clemente with numpy, pandas, scipy, Corrfunc, ...
CODES_DIR=$SLURM_SUBMIT_DIR           # the codes/ folder (submit from there)
# -----------------------------------------------------------------------------------------------------

set -euo pipefail

FIRST=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
LAST=$(( FIRST + CHUNK - 1 ))
(( LAST > N_MEMBERS )) && LAST=$N_MEMBERS
if (( FIRST > N_MEMBERS )); then
    echo "task $SLURM_ARRAY_TASK_ID: nothing to do (FIRST=$FIRST > N_MEMBERS=$N_MEMBERS)"; exit 0
fi

# environment (module system as in the wiki templates; conda from your home on clemente)
source /etc/profile
module load gcc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# OpenMP (Corrfunc) gets the task's cores; numpy/BLAS single-threaded so they don't compete
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# what this task computes (read by the hook in multidark_analysis.py)
export FILBAO_ENS_ONLY=1
export FILBAO_ENS_FIRST=$FIRST
export FILBAO_ENS_LAST=$LAST
export FILBAO_N_ENSEMBLE=$N_MEMBERS

echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): members $FIRST..$LAST," \
     "$SLURM_CPUS_PER_TASK cores, start $(date)"
cd "$CODES_DIR"
srun python -u scripts/multidark_analysis.py
echo "done $(date)"
