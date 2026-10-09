#!/bin/bash
# Evtween training on Rails (Slurm), one GPU. From the cluster (sinfo / scontrol, 2026-10-09):
#   partition gpu      = rails04-06, 8 x H100 80GB (gres gpu:h100), 192 CPUs, ~2 TB RAM per node
#   partition gpu_a100 = rails07-15, 2 x A100 (gres gpu:a100), 112-128 CPUs, 250 GB per node
#   MaxTime 7-00:00:00, DefaultTime 04:00:00 (so --time must be set), DefMemPerCPU 2 GB (so --mem must be set)
#   account beqw-tgirails (the other association, noalloc, has no allocation)
# Defaults below = the earlier interactive allocation (job 283673): 1 x H100, 16 CPUs, 128 GB.
# Flags on the sbatch command line override them, e.g. sbatch --time=7-00:00:00 --mem=256G scripts/train.sbatch ...
#
# usage (from the repository folder, /u/marmary/airlab_v2e/evtween):
#   sbatch scripts/train.sbatch <experiment folder> [run_training.py flags ...]
#   sbatch scripts/train.sbatch runs/teacher_full --stage teacher --steps 200000
# The same command continues an existing experiment: if <experiment>/checkpoints/last.pt exists, --resume is added.
# Ten minutes before the time limit the job asks the run to save last.pt, then requeues itself and resumes.
# A100 instead: sbatch --partition=gpu_a100 --gres=gpu:a100:1 scripts/train.sbatch ...
#SBATCH --job-name=evtween
#SBATCH --account=beqw-tgirails
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
if [ $# -lt 1 ]; then echo "usage: sbatch scripts/train.sbatch <experiment folder> [run_training.py flags]"; exit 2; fi
EXP=$1; shift
cd "${SLURM_SUBMIT_DIR:-.}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) restart ${SLURM_RESTART_COUNT:-0}: $(git branch --show-current) $(git rev-parse --short HEAD)"
RESUME=()
if [ -f "$EXP/checkpoints/last.pt" ]; then RESUME=(--resume); fi
python run_training.py --exp "$EXP" --workers $(( ${SLURM_CPUS_PER_TASK:-10} - 2 )) "${RESUME[@]}" "$@" &
PID=$!
trap 'echo "$(date) time limit near: asking the run to save last.pt"; kill -USR1 $PID' USR1
STATUS=0
wait $PID; STATUS=$?
while kill -0 $PID 2>/dev/null; do wait $PID; STATUS=$?; done   # the trap interrupts wait; wait for the real exit
if [ $STATUS -eq 3 ]; then
    echo "$(date) saved after the signal; requeueing job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
fi
exit $STATUS
# Several GPUs on one node: --gres=gpu:h100:N and replace the python line with
#   torchrun --standalone --nproc_per_node N run_training.py ...
# (torchrun does not pass USR1 to its workers: such jobs rely on the periodic checkpoints; resubmit the same command)
