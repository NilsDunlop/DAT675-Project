#!/bin/bash
#SBATCH --job-name=make_networks
#SBATCH --account=NAISS2025-5-462
#SBATCH --partition=alvis
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -C NOGPU
##SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_make_networks_%j.out

CONTAINER="${SLURM_SUBMIT_DIR}/aev-plig.sif"
cutoff=4

# Generate networks
echo "========================================"
echo "Generating bindingnet graphs"
echo "========================================"
apptainer exec --nv "$CONTAINER" python generate_bindingnet_graphs.py --tag distance-binned #--outdir topology_${cutoff}

echo "========================================"
echo "Generating bindingnet graphs"
echo "========================================"
apptainer exec --nv "$CONTAINER" python generate_bindingdb_graphs.py --tag distance-binned #--outdir topology_${cutoff}

echo "========================================"
echo "Generating bindingnet graphs"
echo "========================================"
apptainer exec --nv "$CONTAINER" python generate_pdbbind_graphs.py --tag distance-binned #--outdir topology_${cutoff}

echo "========================================"
echo "Finished all graphs"
echo "========================================"

echo "========================================"
echo "Generating PyTorch Data"
echo "========================================"
apptainer exec --nv "$CONTAINER" python generate_pytorch_data.py --outdir topology_${cutoff}_processed --input topology_${cutoff}
