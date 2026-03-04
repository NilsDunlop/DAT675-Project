#!/bin/bash
#SBATCH --job-name=aev-plig-test
#SBATCH --account=NAISS2025-5-462
#SBATCH --partition=alvis
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -C NOGPU
##SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm_test_gpu_%j.out

CONTAINER="${SLURM_SUBMIT_DIR}/aev-plig.sif"
echo "========================================">slurm_test_gpu_%j.log
echo "Generating bindingnet graphs">>slurm_test_gpu_%j.log
echo "========================================">>slurm_test_gpu_%j.log
apptainer exec --nv "$CONTAINER" python generate_bindingnet_graphs.py --topology-cutoff 5 > /dev/null 2>>slurm_test_gpu_%j.log


echo "========================================">>slurm_test_gpu_%j.log
echo "Generating bindingnet graphs">>slurm_test_gpu_%j.log
echo "========================================">>slurm_test_gpu_%j.log
apptainer exec --nv "$CONTAINER" python generate_bindingdb_graphs.py --topology-cutoff 5 > /dev/null 2>>slurm_test_gpu_%j.log


echo "========================================">>slurm_test_gpu_%j.log
echo "Generating bindingnet graphs">>slurm_test_gpu_%j.log
echo "========================================">>slurm_test_gpu_%j.log
apptainer exec --nv "$CONTAINER" python generate_pdbbind_graphs.py --topology-cutoff 5 > /dev/null 2>>slurm_test_gpu_%j.log


echo "========================================">>slurm_test_gpu_%j.log
echo "Finished all graphs">>slurm_test_gpu_%j.log
echo "========================================">>slurm_test_gpu_%j.log
