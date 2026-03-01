#!/bin/bash
#SBATCH --job-name=aev-plig-test
#SBATCH --account=NAISS2025-5-462
#SBATCH --partition=alvis
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -C NOGPU
##SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=slurm_test_gpu_%j.out

# Start in the submission directory
cd $SLURM_SUBMIT_DIR

# Define Container Image
CONTAINER="${SLURM_SUBMIT_DIR}/aev-plig.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "Error: Container not found at $CONTAINER"
    exit 1
fi

# Quick GPU sanity check
echo "=== GPU Sanity Check ==="
apptainer exec --nv "$CONTAINER" \
    python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('WARNING: No GPU detected!')
"

# Run example prediction
echo ""
echo "=== Running AEV-PLIG Prediction on Example Dataset ==="
#apptainer exec --nv "$CONTAINER" \
#    python process_and_predict.py \
#        --dataset_csv=data/example_dataset.csv \
#        --data_name=example \
#        --trained_model_name=model_GATv2Net_ligsim90_fep_benchmark


echo ""
echo "=== Generating BindingDB Graphs ==="
apptainer exec --nv "$CONTAINER" python generate_bindingdb_graphs.py

