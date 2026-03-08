#!/bin/bash
#SBATCH --job-name=aev-plig-test
#SBATCH --account=NAISS2025-5-462
#SBATCH --partition=alvis
#SBATCH --nodes=1
#SBATCH -n 1
##SBATCH -C NOGPU
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm_predict_%j.out

# Defaults (can be overridden via flags)
TAG="${TAG:-original}"
MODEL_PREFIX="${MODEL_PREFIX:-}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
TOPOLOGY_CUTOFF=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tag) TAG="$2"; shift ;;
        --model) MODEL_PREFIX="$2"; shift ;;
        --output-prefix) OUTPUT_PREFIX="$2"; shift ;;
        --topology_cutoff) TOPOLOGY_CUTOFF="$2"; shift ;;
    esac
    shift
done

TOPO_FLAG=""
if [ -n "$TOPOLOGY_CUTOFF" ]; then
    TOPO_FLAG="--topology_cutoff=$TOPOLOGY_CUTOFF"
fi

# If no model specified, find the latest trained model for this tag
SUFFIX=$( [ "$TAG" = "original" ] && echo "" || echo "_${TAG}" )
SUFFIX=""
if [ -z "$MODEL_PREFIX" ]; then
    #PATTERN="output/trained_models/*_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark${SUFFIX}_0.model"
    PATTERN="output_cutoff_4_model/trained_models/*_model_GATv2Net_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark${SUFFIX}_0.model"
    LATEST=$(ls -t $PATTERN 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        MODEL_PREFIX=$(basename "$LATEST" | sed 's/_0\.model$//')
    else
        echo "ERROR: No trained model found matching: $PATTERN"
        echo "  Pass --model MODEL_PREFIX explicitly."
        exit 1
    fi
fi

# Output file prefix (defaults to empty, so output is e.g. casf2016_predictions.csv)
OUT=$( [ -z "$OUTPUT_PREFIX" ] && echo "" || echo "${OUTPUT_PREFIX}_" )

echo "========================================"
echo "AEV-PLIG Benchmark Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Tag: $TAG"
echo "Model: $MODEL_PREFIX"
echo "Topology: ${TOPOLOGY_CUTOFF:-bonds (default)}"
echo "Output prefix: ${OUTPUT_PREFIX:-<none>}"
echo "Started: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS on $(hostname)"
echo "========================================"

cd $SLURM_SUBMIT_DIR

mkdir -p output/predictions
mkdir -p slurm/logs

# Common flags (BASE_FLAGS excludes --use_mol2 for datasets without mol2 files, e.g. FEP)
BASE_FLAGS="--trained_model_name=$MODEL_PREFIX --device=0 --num_workers=16 --skip_validation $TOPO_FLAG"
COMMON_FLAGS="$BASE_FLAGS --use_mol2"

# ============================================================
# 1. CASF-2016 (285 complexes)
# ============================================================
echo ""
echo "========================================"
echo "[1/4] CASF-2016 Benchmark (285 complexes)"
echo "Started: $(date)"
echo "========================================"

apptainer exec \
    --bind /mimer/NOBACKUP/groups/naiss2023-6-290:/mimer/NOBACKUP/groups/naiss2023-6-290 \
    --nv aev-plig.sif \
    python process_and_predict.py \
    --dataset_csv="evaluate/casf-2016/casf2016_test.csv" \
    --data_name="${OUT}casf2016${SUFFIX}" \
    $COMMON_FLAGS

echo "CASF-2016 completed: $(date)"

# ============================================================
# 2. 0-LigandBias (366 complexes)
# ============================================================
echo ""
echo "========================================"
echo "[2/4] 0-LigandBias Benchmark (366 complexes)"
echo "Started: $(date)"
echo "========================================"

apptainer exec \
    --bind /mimer/NOBACKUP/groups/naiss2023-6-290:/mimer/NOBACKUP/groups/naiss2023-6-290 \
    --nv aev-plig.sif \
    python process_and_predict.py \
    --dataset_csv="evaluate/0ligandbias/0ligandbias_test.csv" \
    --data_name="${OUT}0ligandbias${SUFFIX}" \
    $COMMON_FLAGS

echo "0-LigandBias completed: $(date)"

# ============================================================
# 3. OOD Test (295 complexes)
# ============================================================
echo ""
echo "========================================"
echo "[3/4] OOD Test Benchmark (295 complexes)"
echo "Started: $(date)"
echo "========================================"

apptainer exec \
    --bind /mimer/NOBACKUP/groups/naiss2023-6-290:/mimer/NOBACKUP/groups/naiss2023-6-290 \
    --nv aev-plig.sif \
    python process_and_predict.py \
    --dataset_csv="evaluate/ood-test/oodtest_test.csv" \
    --data_name="${OUT}oodtest${SUFFIX}" \
    $COMMON_FLAGS

echo "OOD Test completed: $(date)"

# ============================================================
# 4. FEP Benchmark (1185 complexes)
# ============================================================
echo ""
echo "========================================"
echo "[4/4] FEP Benchmark (1185 complexes)"
echo "Started: $(date)"
echo "========================================"

apptainer exec \
    --bind /mimer/NOBACKUP/groups/naiss2023-6-290:/mimer/NOBACKUP/groups/naiss2023-6-290 \
    --nv aev-plig.sif \
    python process_and_predict.py \
    --dataset_csv="evaluate/fep/fep_benchmark_test.csv" \
    --data_name="${OUT}fep${SUFFIX}" \
    $BASE_FLAGS

echo "FEP completed: $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "All benchmarks completed: $(date)"
echo "Predictions saved to output/predictions/"
echo "  - ${OUT}casf2016${SUFFIX}_predictions.csv"
echo "  - ${OUT}0ligandbias${SUFFIX}_predictions.csv"
echo "  - ${OUT}oodtest${SUFFIX}_predictions.csv"
echo "  - ${OUT}fep${SUFFIX}_predictions.csv"
echo "========================================"
