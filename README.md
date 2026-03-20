# AEV-PLIG

AEV-PLIG is a GNN-based scoring function that predicts the binding affinity of a bound protein-ligand complex given its 3D structure. The paper is published in Nature's *Communications Chemistry* at [Narrowing the gap between machine learning scoring functions and free energy perturbation using augmented data](https://doi.org/10.1038/s42004-025-01428-y).

AEV-PLIG was first published in [How to make machine learning scoring functions competitive with FEP](https://chemrxiv.org/engage/chemrxiv/article-details/6675a38d5101a2ffa8274f62), and received the [people's poster prize at the 7th AI in Chemistry Symposium](https://www.stats.ox.ac.uk/news/isak-valsson-wins-poster-prize). In the paper we benchmark AEV-PLIG on a wide range of benchmarks, including CASF-2016, our new out-of-distribution benchmark OOD Test, and a test set used for free energy perturbation (FEP) calculations, and highlight competitive performance across the board. Moreover, we demonstrate how leveraging augmented data (generated using template-based modelling or molecular docking) can significantly improve binding affinity prediction correlation and ranking on the FEP benchmark (PCC and Kendall's increases from 0.41 and 0.26, to 0.59 and 0.42), closing the performance gap with FEP calculations while being 400,000 times faster.

## Our extensions

In this fork we investigate how two independent modifications affect AEV-PLIG performance:

**Topology modifications** — The original model uses covalent bond-based graph edges (single, aromatic, double, triple). We replace these with distance-cutoff spatial edges, where any two heavy atoms within a configurable radius (e.g. 2, 4, 5, 6, 50 A) are connected with edge weight equal to their Euclidean distance. This is controlled by the `--topology-cutoff` argument.

**Feature encoding modifications** — The original AEV node features use 16 radial Gaussian shifts. We experiment with alternative encoding schemes:
- `binary` — binary presence/absence of each protein atom type within the cutoff
- `distance-binned` — atom-type counts within concentric distance shells
- `reduced-gaussian-4` — 4 evenly-spaced radial Gaussian shifts
- `reduced-gaussian-8` — 8 evenly-spaced radial Gaussian shifts

These are controlled by the `--encoding` argument. Both modifications can be used independently or combined.

## Table of contents

- [Data](#data)
- [Installation](#installation)
- [Training pipeline](#training-pipeline)
- [Predictions](#predictions)
- [Evaluation](#evaluation)
- [AlphaFold 3 comparison](#alphafold-3-comparison)
- [SLURM (HPC)](#slurm-hpc)
- [Project structure](#project-structure)

## Data

All required data (PDBbind, BindingNet, BindingDB-DCS, benchmark test sets, pre-trained models) is available at:

**[Download data](https://chalmers-my.sharepoint.com/:f:/g/personal/nilsdu_chalmers_se/IgC_GxtHrw_nQoldgXJoz9T_AWij0tmU0WE8q0egdIAvv00?e=OhJnmA)**

After downloading, place the data so the directory layout matches:

```
data/
  pdbbind/refined-set/       # PDBbind v2020 refined
  pdbbind/general-set/       # PDBbind v2020 general
  bindingnet/from_chembl_client/
  bindingdb/surflex/
  pdbbind_processed.csv
  bindingnet_processed.csv
  bindingdb_processed.csv
  PDB_Atom_Keys.csv
evaluate/
  casf-2016/casf2016_test.csv
  0ligandbias/0ligandbias_test.csv
  ood-test/oodtest_test.csv
  fep/fep_benchmark_test.csv
```

## Installation

AEV-PLIG has been tested on macOS (Monterey 12.5.1) and Linux (Ubuntu 22.04.5 LTS).

### Conda (recommended)

For macOS:
```bash
conda env create --file aev-plig-mac.yml
```

For Linux:
```bash
conda env create --file aev-plig-linux.yml
```

### Manual install

```bash
conda create --name aev-plig python=3.8
conda activate aev-plig
pip install torch torchvision torchaudio torch-scatter torch_geometric rdkit torchani qcelemental pandas biopandas scikit-learn scipy
```

## Training pipeline

Training follows three steps: graph generation, PyTorch data creation, and model training.

### 1. Generate graphs

Build protein-ligand interaction graphs as pickle files. Both `--topology-cutoff` and `--encoding` can be configured.

```bash
python generate_pdbbind_graphs.py   --topology-cutoff 5.0 --outdir cutoff5 --encoding original
python generate_bindingnet_graphs.py --topology-cutoff 5.0 --outdir cutoff5 --encoding original
python generate_bindingdb_graphs.py  --topology-cutoff 5.0 --outdir cutoff5 --encoding original
```

This produces `data/cutoff5/pdbbind.pickle`, `bindingnet.pickle`, and `bindingdb.pickle`.

| Argument | Description |
|---|---|
| `--topology-cutoff` | Distance cutoff in Angstroms for spatial graph edges |
| `--outdir` | Output subdirectory under `data/` for the pickle files |
| `--encoding` | AEV encoding: `original`, `binary`, `distance-binned`, `reduced-gaussian-4`, `reduced-gaussian-8` |

**SLURM:** All three datasets run as a SLURM array job (see [SLURM scripts](#slurm-hpc)):

```bash
sbatch slurm/generate_graphs.sh
```

### 2. Create PyTorch data

Convert the pickle graphs into `.pt` datasets, applying benchmark test set exclusion.

```bash
python create_pytorch_data.py --tag cutoff5 --encoding original
```

The `--tag` argument is a convenience shorthand that sets `--graph_dir` to `cutoff5` and `--outdir` to `processed_cutoff5`. The resulting `.pt` files are written to `data/processed_cutoff5/`.

| Argument | Description |
|---|---|
| `--tag` | Variant tag — auto-sets `--outdir` to `processed_{tag}` and `--graph_dir` to `{tag}` |
| `--encoding` | AEV encoding (appended to dataset name when not `original`) |
| `--graph_dir` | Subdirectory under `data/` containing pickle files (overrides `--tag`) |
| `--outdir` | Subdirectory under `data/` for `.pt` output (overrides `--tag`) |
| `--skip_exclusion` | Skip benchmark test ID exclusion (causes data leakage) |

**SLURM:**

```bash
sbatch slurm/generate_pytorch_data.sh
```

### 3. Train model

Train a 10-seed GATv2Net ensemble. Requires a GPU (~28h on NVIDIA GTX 1080 Ti).

```bash
python training.py \
    --model=GATv2Net \
    --dataset=pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark \
    --input=processed_cutoff5 \
    --output=output_cutoff5 \
    --batch_size=128 \
    --epochs=200 \
    --hidden_dim=256 \
    --head=3 \
    --lr=0.00012 \
    --activation_function=leaky_relu
```

Trained models are saved to `{output}/trained_models/`.

| Argument | Description |
|---|---|
| `--input` | Subdirectory under `data/` with the `.pt` files (must match `create_pytorch_data.py --outdir`) |
| `--output` | Output directory for trained model files |
| `--dataset` | Dataset name prefix (must match the `.pt` filenames without `_train/_valid/_test.pt`) |
| `--tag` | Label for WandB run names |
| `--wandb_project` | WandB project name |

**SLURM:**

```bash
sbatch slurm/train_model.sh
```

## Predictions

To run AEV-PLIG on new data, provide a CSV with columns `unique_id`, `sdf_file`, and `pdb_file`:

```bash
python process_and_predict.py \
    --dataset_csv=data/example_dataset.csv \
    --data_name=example \
    --trained_model_name=model_GATv2Net_ligsim90_fep_benchmark \
    --encoding=original \
    --topology_cutoff=5.0 \
    --model_dir=output_cutoff5/trained_models \
    --num_models=10
```

The script validates molecules (RDKit readability, rare elements, undefined bonds), generates graphs, and runs the ensemble. Predictions are saved to `output/predictions/{data_name}_predictions.csv`.

| Argument | Description |
|---|---|
| `--dataset_csv` | Input CSV with `unique_id`, `sdf_file`, `pdb_file` columns |
| `--data_name` | Name for output prediction file |
| `--trained_model_name` | Model filename prefix in `--model_dir` |
| `--model_dir` | Directory containing `.model` and `.pickle` files |
| `--encoding` | Must match the encoding used during training |
| `--topology_cutoff` | Must match the topology cutoff used during training |
| `--num_models` | Number of ensemble models (default: 10) |
| `--use_mol2` | Load ligands from `.mol2` instead of `.sdf` |
| `--skip_validation` | Skip Biopandas protein validation |

## Evaluation

### Running predictions on benchmarks

Run `process_and_predict.py` for each benchmark test set (CASF-2016, 0-LigandBias, OOD Test, FEP):

```bash
for benchmark in casf2016 0ligandbias oodtest fep; do
    python process_and_predict.py \
        --dataset_csv=evaluate/${benchmark}/${benchmark}_test.csv \
        --data_name=${benchmark}_cutoff5 \
        --trained_model_name=<YOUR_MODEL_NAME> \
        --model_dir=output_cutoff5/trained_models \
        --topology_cutoff=5.0 \
        --encoding=original \
        --num_models=10
done
```

Adjust `evaluate/${benchmark}/` paths as needed (e.g. `evaluate/casf-2016/casf2016_test.csv`, `evaluate/0ligandbias/0ligandbias_test.csv`, `evaluate/ood-test/oodtest_test.csv`, `evaluate/fep/fep_benchmark_test.csv`).

**SLURM:** Run all four benchmarks in a single job. The script auto-discovers the latest trained model for the given `--tag`:

```bash
sbatch slurm/eval_benchmarks.sh --tag original
```

You can also pass `--model MODEL_PREFIX` to specify a model explicitly.

### Comparing against baselines

Compare your predictions against the original authors' published results:

```bash
python compare_baselines.py --tag cutoff5
```

This computes PCC, Kendall tau, and RMSE for each benchmark, and compares against the paper's reported metrics (Table 1). Prediction CSVs are expected at `output/predictions/{benchmark}_{tag}_predictions.csv`.

**SLURM:**

```bash
sbatch slurm/compare_baselines.sh --tag cutoff5
```

### Metrics

| Benchmark | Metrics |
|---|---|
| CASF-2016 | PCC, Kendall tau, RMSE |
| 0-LigandBias | PCC, Kendall tau, RMSE |
| OOD Test | PCC, Kendall tau, RMSE |
| FEP Benchmark | Weighted-mean PCC and Kendall tau per ligand series, RMSE |

### Plotting

Generate correlation scatter plots:

```bash
python figures/plot_model_scatter.py
```

## AlphaFold 3 comparison

The `af3/` directory contains scripts for comparing AEV-PLIG predictions on AlphaFold 3-generated structures versus experimental structures.

```bash
# 1. Generate AF3 input JSONs from benchmark CSVs
python af3/generate_af3_inputs.py --benchmark_csv=evaluate/casf-2016/casf2016_test.csv --benchmark_name=casf2016

# 2. Process AF3 CIF outputs into PDB + SDF files for AEV-PLIG
python af3/process_af3_outputs.py --af3_dir=af3/outputs --output_dir=af3/processed --benchmark=casf2016

# 3. Compute RMSD between AF3 and experimental structures
python af3/compute_af3_rmsd.py --af3_dir=af3/processed --eval_dir=evaluate --benchmarks casf2016

# 4. Compare AEV-PLIG predictions on AF3 vs experimental structures
python af3/compare_af3_vs_experimental.py --tag cutoff5
```

**SLURM:** Evaluate all three benchmarks (CASF-2016, 0-LigandBias, OOD Test) on AF3-predicted structures:

```bash
sbatch slurm/eval_af3_benchmarks.sh --tag original
```

## SLURM (HPC)

The `slurm/` directory contains ready-to-use SLURM scripts for each pipeline step. All scripts use Apptainer with an `aev-plig.sif` container and accept a `--tag` argument to select the model variant. Logs are written to `slurm/logs/`.

```bash
mkdir -p slurm/logs
```

### Full pipeline

Run each step as a separate job, chaining with dependencies:

```bash
# 1. Generate graphs (array job: 3 tasks, one per dataset, ~24h CPU)
JOB1=$(sbatch --parsable slurm/generate_graphs.sh)

# 2. Create PyTorch data (after graphs finish, ~24h CPU)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/generate_pytorch_data.sh)

# 3. Train model (after data is ready, ~24h GPU)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 slurm/train_model.sh)

# 4. Evaluate on all benchmarks (after training, ~6h GPU)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 slurm/eval_benchmarks.sh)

# 5. Compare against author baselines (~10 min CPU)
sbatch --dependency=afterok:$JOB4 slurm/compare_baselines.sh
```

All scripts inherit `--tag` to keep the variant consistent. Set it via environment variable or argument:

```bash
TAG=binary sbatch slurm/generate_graphs.sh
# or
sbatch slurm/generate_graphs.sh --tag binary
```

### Available scripts

| Script | Resources | Description |
|---|---|---|
| `slurm/generate_graphs.sh` | CPU, 16 cores, 64 GB, array(0-2) | Runs `generate_{pdbbind,bindingnet,bindingdb}_graphs.py` as parallel array tasks |
| `slurm/generate_pytorch_data.sh` | CPU, 16 cores, 128 GB | Runs `create_pytorch_data.py`, clears stale `.pt` caches |
| `slurm/train_model.sh` | 1x A100 GPU, 16 cores, 64 GB | Runs `training.py` with WandB logging (reads key from `.env`) |
| `slurm/eval_benchmarks.sh` | 1x A100 GPU, 16 cores, 64 GB | Evaluates on CASF-2016, 0-LigandBias, OOD Test, and FEP |
| `slurm/eval_af3_benchmarks.sh` | 1x A40 GPU, 16 cores, 64 GB | Evaluates on AF3-predicted structures (3 benchmarks) |
| `slurm/eval_model.sh` | 1x A100 GPU, 16 cores, 64 GB | Single prediction run on any CSV dataset |
| `slurm/compare_baselines.sh` | CPU, 4 cores, 8 GB | Runs `compare_baselines.py` to compare against paper metrics |

### Legacy templates

Two standalone SLURM templates from the topology branch are also available in the project root:

- `run_full_process_template.slurm` — Combined training job. Replace `XXXTOPOLOGYCUTOFFXXX` with the cutoff value: `sed 's/XXXTOPOLOGYCUTOFFXXX/5/g' run_full_process_template.slurm > run_cutoff5.slurm`
- `make_networks.slurm` — Combined graph generation job

### WandB setup

Training scripts log to Weights & Biases. Store your API key in a `.env` file in the project root:

```
WANDB_API_KEY=your_key_here
```

## Project structure

```
.
├── training.py                   # Model training (10-seed ensemble)
├── create_pytorch_data.py        # Pickle graphs -> PyTorch .pt datasets
├── generate_pdbbind_graphs.py    # PDBbind graph generation
├── generate_bindingnet_graphs.py # BindingNet graph generation
├── generate_bindingdb_graphs.py  # BindingDB graph generation
├── process_and_predict.py        # End-to-end prediction pipeline
├── compare_baselines.py          # Compare predictions vs author baselines
├── model_defs.py                 # GATv2Net model definition
├── helpers.py                    # Metrics (RMSE, PCC, CI) and model registry
├── utils.py                      # GraphDataset, weight init, scalers
├── torchani_mod/                 # Modified TorchANI for AEV computation
├── data/                         # Datasets, processed CSVs, pickle graphs
│   ├── unify_data.py             # Dataset merging and standardization
│   └── streamline_fep.py         # FEP benchmark CSV builder
├── evaluate/                     # Benchmark test CSVs
│   ├── casf-2016/
│   ├── 0ligandbias/
│   ├── ood-test/
│   └── fep/
├── af3/                          # AlphaFold 3 comparison pipeline
├── figures/                      # Plotting scripts and output figures
├── output/                       # Trained models and predictions
├── slurm/                        # SLURM job scripts for HPC
│   ├── generate_graphs.sh        # Array job: graph generation (3 datasets)
│   ├── generate_pytorch_data.sh  # PyTorch .pt dataset creation
│   ├── train_model.sh            # Model training with WandB
│   ├── eval_benchmarks.sh        # Evaluate on all 4 benchmarks
│   ├── eval_af3_benchmarks.sh    # Evaluate on AF3-predicted structures
│   ├── eval_model.sh             # Single dataset prediction
│   └── compare_baselines.sh      # Baseline comparison
├── run_full_process_template.slurm  # Legacy: combined training template
├── make_networks.slurm              # Legacy: combined graph generation
├── aev-plig-linux.yml            # Conda environment (Linux)
├── aev-plig-mac.yml              # Conda environment (macOS)
└── README.md
```
