#!/bin/bash
# =====================================================================================================
# FilBAO - dilution ensemble on Clemente (IATE): random dilutions of Full, monopoles only.
#
# A SLURM job array. The members j = 1..N_MEMBERS are cut into chunks of CHUNK members; each array task
# runs PER_JOB chunks side by side (one python process per chunk, 16 Corrfunc threads each), because
# this account may run only 2 jobs at a time (MaxJobs = 2): each job then uses most of a node.
# Each process computes its members for every dilution size of the run in scripts/multidark_analysis.py
# (3 sizes in the cumulative run). Member j has seed dilution_seed + n_dilution_realisations + j - 1
# (20261016 + j now) and writes ..._FullrandomN-<N>r<seed>.npz, exactly as n_dilution_ensemble does on
# the laptop. Members that already have their pair-count file are only reloaded (seconds), so a task that
# crashes or times out can be resubmitted: sbatch --array=<ids> ...
# Ensemble = the original dilution + N_MEMBERS seeds = 1 + 499 = 500.
#
# Needs: the FILBAO_ENS_* hook in scripts/multidark_analysis.py (ensemble-only mode), and Corrfunc in the
# conda env compiled on clemente with gcc/14.3.0 (OpenMP + AVX2; a pip wheel built with the old compiler
# ran on one thread, ~15x slower).
#
# Use, on clemente, from the codes/ folder (the one with scripts/, src/, data/):
#   mkdir -p logs
#   sbatch scripts/run_dilution_ensemble.sbatch
#   squeue -u $USER ;  tail logs/dil_<job>_<task>_p<k>.out
# =====================================================================================================
#SBATCH --job-name=filbao-dil
#SBATCH --partition=batch            # 3-day limit, clemente01-06
#SBATCH --time=12:00:00              # per task (3 chunks in parallel, 10 members x 3 sizes each)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48           # PER_JOB x 16 Corrfunc threads
#SBATCH --array=0-16%2               # 17 tasks x 3 chunks = 51 chunks >= ceil(499 / 10); 2 = account limit
#SBATCH --output=logs/dil_%A_%a.out
#SBATCH --error=logs/dil_%A_%a.err
# (no --mem: the nodes report 1 MB to SLURM, so any memory request is refused; each has 128 GB)

# ---------------- settings ---------------------------------------------------------------------------
N_MEMBERS=499                         # members j = 1..N_MEMBERS (plus the original dilution = 500)
CHUNK=10                              # members per python process
PER_JOB=3                             # processes per array task; --array must cover 0..ceil(N_MEMBERS/(CHUNK*PER_JOB))-1
THREADS=16                            # Corrfunc threads per process (multidark_correlation uses min(cores - 4, 16))
CONDA_ENV=filbao2                     # env on clemente with numpy, pandas, scipy, Corrfunc (built with gcc 14.3.0)
GCC_MODULE=gcc/14.3.0                 # the compiler Corrfunc was built with (its OpenMP runtime is needed)
CODES_DIR=$SLURM_SUBMIT_DIR           # the codes/ folder (submit from there)
# -----------------------------------------------------------------------------------------------------

set -eo pipefail                      # no -u: /etc/profile and conda activate use unset variables

# environment
source /etc/profile
module load "$GCC_MODULE"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export FILBAO_ENS_ONLY=1
export FILBAO_N_ENSEMBLE=$N_MEMBERS

cd "$CODES_DIR"
echo "job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): $SLURM_CPUS_PER_TASK cores, start $(date)"

# one python process per chunk, in the background; each writes its own log
pids=()
for (( k = 0; k < PER_JOB; k++ )); do
    c=$(( SLURM_ARRAY_TASK_ID * PER_JOB + k ))              # chunk number
    first=$(( c * CHUNK + 1 ))
    last=$(( first + CHUNK - 1 ))
    (( last > N_MEMBERS )) && last=$N_MEMBERS
    (( first > N_MEMBERS )) && continue
    log=logs/dil_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_p${k}
    echo "  process $k: members $first..$last -> $log.out"
    FILBAO_ENS_FIRST=$first FILBAO_ENS_LAST=$last python -u scripts/multidark_analysis.py > "$log.out" 2> "$log.err" &
    pids+=($!)
done

# wait for all of them; the task fails if any process failed
status=0
for p in "${pids[@]}"; do
    wait "$p" || status=1
done
echo "done $(date), status $status"
exit $status
