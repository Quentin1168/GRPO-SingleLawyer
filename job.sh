#!/bin/bash
#SBATCH --job-name=grpo-lawyer
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4          # node has 8; 4 schedules easier, node is exclusive anyway
#SBATCH --time=48:00:00            # EXPLICIT — DefaultTime on this partition is a 1-hour trap
#SBATCH --requeue                  # QoS is "spot": if the VM is reclaimed, auto-resubmit
#SBATCH --output=slurm_%j.log
# NOTE: no --mem on purpose — memory is unscheduled on this cluster (RealMemory=1)

set -euo pipefail

PROJECT=$HOME/projects/GRPO-SingleLawyer

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

apptainer exec --nv   \
	--bind $PWD:/workspace   \
	--env HF_HOME=$HOME/hf_cache   \
	--env ART_SERVER_TIMEOUT=900   \
	--env no_proxy=localhost,127.0.0.1,0.0.0.0   \
	--env NO_PROXY=localhost,127.0.0.1,0.0.0.0   \
	--env PYTORCH_ALLOC_CONF=expandable_segments:True   \
	--env RUN_NAME=$RUN_NAME \
	--pwd /workspace   \
	grpo.sif python3 ART_train.py

echo "=== Finished at $(date) ==="
Submit and watch:


sbatch job.sh
squeue -u $USER                              # ST column: PD=queued, R=running
tail -f train_<jobid>.log
