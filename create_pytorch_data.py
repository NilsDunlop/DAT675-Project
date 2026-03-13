import pandas as pd
import pickle
import argparse
from pathlib import Path
from utils import GraphDataset


def load_benchmark_test_ids():
    """Load all benchmark test set PDB codes that must be excluded from training.

    Returns the set of PDB codes (lowercase) appearing in any benchmark test CSV.
    """
    casf = set(pd.read_csv("evaluate/casf-2016/casf2016_test.csv")["unique_id"])
    ligandbias = set(pd.read_csv("evaluate/0ligandbias/0ligandbias_test.csv")["unique_id"])
    oodtest = set(pd.read_csv("evaluate/ood-test/oodtest_test.csv")["unique_id"])

    all_test_ids = casf | ligandbias | oodtest
    all_pdb_codes = set(pid.lower() for pid in all_test_ids)

    print("Benchmark test IDs to exclude from training:")
    print(f"  CASF-2016:    {len(casf)}")
    print(f"  0-LigandBias: {len(ligandbias)}")
    print(f"  OOD Test:     {len(oodtest)}")
    print(f"  Total unique: {len(all_test_ids)}")

    return all_test_ids, all_pdb_codes


def main():
    parser = argparse.ArgumentParser(description="Process graph datasets for model training.")
    parser.add_argument('--outdir', type=str, default=None,
                        help='Subdirectory under data/ for .pt output (must match training.py --input)')
    parser.add_argument('--graph_dir', type=str, default=None,
                        help='Subdirectory under data/ containing .pickle graph files (default: data/ root)')
    parser.add_argument('--tag', type=str, default=None,
                        help='Variant tag (e.g. cutoff4, binary). '
                             'Sets --outdir to processed_{tag} and --graph_dir to {tag} '
                             'unless those are explicitly provided.')
    parser.add_argument('--skip_exclusion', action='store_true',
                        help='Skip benchmark exclusion (WARNING: causes data leakage)')
    args = parser.parse_args()

    if args.tag is not None:
        if args.outdir is None:
            args.outdir = f'processed_{args.tag}'
        if args.graph_dir is None:
            args.graph_dir = args.tag
    if args.outdir is None:
        args.outdir = 'processed'
    if args.graph_dir is None:
        args.graph_dir = ''

    outdir = args.outdir
    graph_base = Path("data") / args.graph_dir if args.graph_dir else Path("data")
    pt_dir = Path("data") / outdir
    pt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load graphs
    # ------------------------------------------------------------------
    print("Loading graph pickle files...")
    with open(graph_base / "pdbbind.pickle", 'rb') as handle:
        pdbbind_graphs = pickle.load(handle)
    print(f"  pdbbind:    {len(pdbbind_graphs)} graphs")

    with open(graph_base / "bindingnet.pickle", 'rb') as handle:
        bindingnet_graphs = pickle.load(handle)
    print(f"  bindingnet: {len(bindingnet_graphs)} graphs")

    with open(graph_base / "bindingdb.pickle", 'rb') as handle:
        bindingdb_graphs = pickle.load(handle)
    print(f"  bindingdb:  {len(bindingdb_graphs)} graphs")

    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}

    # ------------------------------------------------------------------
    # Load benchmark test IDs for exclusion
    # ------------------------------------------------------------------
    if not args.skip_exclusion:
        exclude_ids, exclude_pdb_codes = load_benchmark_test_ids()
    else:
        print("WARNING: Skipping benchmark exclusion — training data may leak into test sets!")
        exclude_ids = set()
        exclude_pdb_codes = set()

    # ------------------------------------------------------------------
    # PDBbind
    # ------------------------------------------------------------------
    pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
    pdbbind = pdbbind[['PDB_code', '-logKd/Ki', 'split_core', 'max_tanimoto_fep_benchmark']]
    pdbbind = pdbbind.rename(columns={'PDB_code': 'unique_id', 'split_core': 'split', '-logKd/Ki': 'pK'})
    pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]

    if len(exclude_ids) > 0:
        train_before = len(pdbbind[pdbbind['split'] == 'train'])
        valid_before = len(pdbbind[pdbbind['split'] == 'valid'])

        pdbbind = pdbbind[~((pdbbind['split'].isin(['train', 'valid'])) &
                            (pdbbind['unique_id'].str.lower().isin(exclude_pdb_codes)))]

        train_after = len(pdbbind[pdbbind['split'] == 'train'])
        valid_after = len(pdbbind[pdbbind['split'] == 'valid'])
        print(f"\nPDBbind exclusion:")
        print(f"  Train: {train_before} -> {train_after} (removed {train_before - train_after})")
        print(f"  Valid: {valid_before} -> {valid_after} (removed {valid_before - valid_after})")
        print(f"  Test:  {len(pdbbind[pdbbind['split'] == 'test'])} (unchanged)")

    pdbbind = pdbbind[['unique_id', 'pK', 'split']]

    # ------------------------------------------------------------------
    # BindingNet — exclude entries whose PDB code appears in any benchmark
    # ------------------------------------------------------------------
    bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
    bindingnet = bindingnet.rename(columns={'-logAffi': 'pK', 'unique_identify': 'unique_id'})
    bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.9]

    if len(exclude_pdb_codes) > 0:
        bn_before = len(bindingnet)
        bindingnet = bindingnet[~bindingnet['pdb'].str.lower().isin(exclude_pdb_codes)]
        bn_after = len(bindingnet)
        print(f"\nBindingNet exclusion:")
        print(f"  Removed {bn_before - bn_after} entries with PDB codes in benchmark test sets")

    bindingnet['split'] = 'train'
    bindingnet = bindingnet[['unique_id', 'pK', 'split']]

    # ------------------------------------------------------------------
    # BindingDB — exclude entries whose PDB code appears in any benchmark
    # ------------------------------------------------------------------
    bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
    bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9]

    if len(exclude_pdb_codes) > 0:
        bdb_before = len(bindingdb)
        bdb_pdb_codes = bindingdb['pdb_file'].str.replace('.pdb', '', regex=False).str.lower()
        bindingdb = bindingdb[~bdb_pdb_codes.isin(exclude_pdb_codes)]
        bdb_after = len(bindingdb)
        print(f"\nBindingDB exclusion:")
        print(f"  Removed {bdb_before - bdb_after} entries with PDB codes in benchmark test sets")

    bindingdb['split'] = 'train'
    bindingdb = bindingdb[['unique_id', 'pK', 'split']]

    # ------------------------------------------------------------------
    # Combine and filter to entries with available graphs
    # ------------------------------------------------------------------
    data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)

    available = set(graphs_dict.keys())
    before_graph_filter = len(data)
    data = data[data['unique_id'].isin(available)]
    after_graph_filter = len(data)
    if before_graph_filter != after_graph_filter:
        print(f"\nNote: {before_graph_filter - after_graph_filter} entries skipped (no graph in pickle)")

    print(f"\nFinal split counts:")
    print(data[['split']].value_counts())

    dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'
    if args.tag:
        dataset = f'{dataset}_{args.tag}'

    df_train = data[data['split'] == 'train']
    train_ids, train_y = list(df_train['unique_id']), list(df_train['pK'])

    df_valid = data[data['split'] == 'valid']
    valid_ids, valid_y = list(df_valid['unique_id']), list(df_valid['pK'])

    df_test = data[data['split'] == 'test']
    test_ids, test_y = list(df_test['unique_id']), list(df_test['pK'])

    # ------------------------------------------------------------------
    # Build PyTorch Geometric datasets
    # ------------------------------------------------------------------
    print(f"\nPreparing {dataset}_train.pt ...")
    train_data = GraphDataset(root='data', outdir=outdir, dataset=dataset + '_train',
                              ids=train_ids, y=train_y, graphs_dict=graphs_dict)

    print(f"Preparing {dataset}_valid.pt ...")
    valid_data = GraphDataset(root='data', outdir=outdir, dataset=dataset + '_valid',
                              ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

    print(f"Preparing {dataset}_test.pt ...")
    test_data = GraphDataset(root='data', outdir=outdir, dataset=dataset + '_test',
                             ids=test_ids, y=test_y, graphs_dict=graphs_dict)

    print(f"\nDone. Files written to data/{outdir}/")


if __name__ == "__main__":
    main()
