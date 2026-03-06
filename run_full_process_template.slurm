#!/bin/bash
#SBATCH --job-name=aev-plig-test
#SBATCH --account=NAISS2025-5-462
#SBATCH --partition=alvis
#SBATCH --nodes=1
#SBATCH -n 1
##SBATCH -C NOGPU
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=slurm_topology_%j.out

cutoff=XXXCUTOFFXXX

echo "========================================"
echo "Training AEV-PLIG Model"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS on $(hostname)"
echo "topology-cutoff: ${cutoff}"
echo "========================================"
CONTAINER="${SLURM_SUBMIT_DIR}/aev-plig.sif"

cd $SLURM_SUBMIT_DIR

# Ensure output directory exists before training
mkdir -p output_cutoff_${cutoff}_models

# Extract WANDB_API_KEY from .env
if [ -f ".env" ]; then
    export WANDB_API_KEY=$(grep '^WANDB_API_KEY=' .env | cut -d '=' -f2)
else
    echo "Warning: .env file not found. If WandB requires a key it might prompt or fail."
fi

# Configure WandB specific settings for HPC
mkdir "${SLURM_SUBMIT_DIR}/wandb/cutoff_${cutoff}"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/wandb/cutoff_${cutoff}/wandb_logs"
export WANDB_CACHE_DIR="${SLURM_SUBMIT_DIR}/wandb/cutoff_${cutoff}/wandb_cache"
export WANDB_CONFIG_DIR="${SLURM_SUBMIT_DIR}/wandb/cutoff_${cutoff}/wandb_config"

mkdir -p $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR

# Run the training script via the apptainer container
# Binding the project directory is required for file access
# We use --nv to enable Nvidia GPU support inside the container
apptainer exec \
    --bind /mimer/NOBACKUP/groups/naiss2023-6-290:/mimer/NOBACKUP/groups/naiss2023-6-290 \
    --nv \
    --env PYTHONUNBUFFERED=1 \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env WANDB_DIR="$WANDB_DIR" \
    --env WANDB_CACHE_DIR="$WANDB_CACHE_DIR" \
    --env WANDB_CONFIG_DIR="$WANDB_CONFIG_DIR" \
    $CONTAINER \
    python training.py \
    --activation_function=leaky_relu \
    --batch_size=128 \
    --dataset=pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark \
    --epochs=200 \
    --head=3 \
    --hidden_dim=256 \
    --lr=0.00012 \
    --model=GATv2Net \
    --input=topology_${cutoff}_processed\
    --output=output_cutoff_${cutoff}_model

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
