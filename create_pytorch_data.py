import pandas as pd
import pickle
import argparse
from utils import GraphDataset
import argparse

def load_benchmark_test_ids(input_dir: Path):
    """Load all benchmark test set IDs that must be excluded from training."""
    eval_dir = input_dir / "evaluate"
    
    casf = set(pd.read_csv(eval_dir / "casf-2016" / "casf2016_test.csv")["unique_id"])
    ligandbias = set(pd.read_csv(eval_dir / "0ligandbias" / "0ligandbias_test.csv")["unique_id"])
    oodtest = set(pd.read_csv(eval_dir / "ood-test" / "oodtest_test.csv")["unique_id"])
    
    all_test_ids = casf | ligandbias | oodtest
    
    print(f"Benchmark test IDs to exclude from training:")
    print(f"CASF-2016: {len(casf)}")
    print(f"0-LigandBias: {len(ligandbias)}")
    print(f"OOD Test: {len(oodtest)}")
    print(f"Total unique: {len(all_test_ids)}")
    
    return all_test_ids


def main():
    parser = argparse.ArgumentParser(description="Process graph datasets for model training.")
    parser.add_argument('--outdir', type=str, default='processed', 
                        help='Output directory for the processed datasets')
    parser.add_argument('--input', type=str, default='', 
                        help='Base input directory containing data and evaluate folders')
    parser.add_argument('--skip_exclusion', action='store_true',
                        help='Skip benchmark exclusion')
    args = parser.parse_args()

    input_dir = Path(args.input)
    outdir = Path(args.outdir)
    data_dir = input_dir / "data"
    outdir.mkdir(parents=True, exist_ok=True)

    """
    Load graphs
    """
    print("loading graph from pickle file for pdbbind2020")
    with open(data_dir / "pdbbind.pickle", 'rb') as handle:
        pdbbind_graphs = pickle.load(handle)

    print("loading graph from pickle file for BindingNet")
    with open(data_dir / "bindingnet.pickle", 'rb') as handle:
        bindingnet_graphs = pickle.load(handle)

    print("loading graph from pickle file for BindingDB")
    with open(data_dir / "bindingdb.pickle", 'rb') as handle:
        bindingdb_graphs = pickle.load(handle)

    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}

    """
    Load benchmark test IDs for exclusion
    """
    if not args.skip_exclusion:
        exclude_ids = load_benchmark_test_ids(input_dir)
    else:
        print("WARNING: Skipping benchmark exclusion — training data may leak into test sets!")
        exclude_ids = set()

    """
    Generate data for enriched training for <0.9 Tanimoto to the FEP benchmark
    """
    pdbbind = pd.read_csv(data_dir / "pdbbind_processed.csv", index_col=0)
    pdbbind = pdbbind[['PDB_code', '-logKd/Ki', 'split_core', 'max_tanimoto_fep_benchmark']]
    pdbbind = pdbbind.rename(columns={'PDB_code': 'unique_id', 'split_core': 'split', '-logKd/Ki': 'pK'})
    pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]
    pdbbind = pdbbind[['unique_id', 'pK', 'split']]

    # Exclude benchmark test IDs from train/valid splits
    if len(exclude_ids) > 0:
        train_before = len(pdbbind[pdbbind['split'] == 'train'])
        valid_before = len(pdbbind[pdbbind['split'] == 'valid'])
        
        pdbbind = pdbbind[~((pdbbind['split'].isin(['train', 'valid'])) & 
                            (pdbbind['unique_id'].isin(exclude_ids)))]
        
        train_after = len(pdbbind[pdbbind['split'] == 'train'])
        valid_after = len(pdbbind[pdbbind['split'] == 'valid'])
        print(f"\nPDBbind exclusion results:")
        print(f"  Train: {train_before} -> {train_after} (removed {train_before - train_after})")
        print(f"  Valid: {valid_before} -> {valid_after} (removed {valid_before - valid_after})")
        print(f"  Test:  {len(pdbbind[pdbbind['split'] == 'test'])} (unchanged)")

    bindingnet = pd.read_csv(data_dir / "bindingnet_processed.csv", index_col=0)
    bindingnet = bindingnet.rename(columns={'-logAffi': 'pK', 'unique_identify': 'unique_id'})[['unique_id', 'pK', 'max_tanimoto_fep_benchmark']]
    bindingnet['split'] = 'train'
    bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.9]
    bindingnet = bindingnet[['unique_id', 'pK', 'split']]

    bindingdb = pd.read_csv(data_dir / "bindingdb_processed.csv", index_col=0)
    bindingdb = bindingdb[['unique_id', 'pK', 'max_tanimoto_fep_benchmark']]
    bindingdb['split'] = 'train'
    bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9]
    bindingdb = bindingdb[['unique_id', 'pK', 'split']]

    # combine pdbbind2020, bindingnet, and bindingdb index sets
    data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
    print(f"\nFinal split counts:")
    print(data[['split']].value_counts())

    dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'

    df = data[data['split'] == 'train']
    train_ids, train_y = list(df['unique_id']), list(df['pK'])

    df = data[data['split'] == 'valid']
    valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

    df = data[data['split'] == 'test']
    test_ids, test_y = list(df['unique_id']), list(df['pK'])

    # make data PyTorch Geometric ready
    print(f"preparing {dataset}_train.pt in pytorch format!")
    train_data = GraphDataset(root=str(outdir), dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

    print(f"preparing {dataset}_valid.pt in pytorch format!")
    valid_data = GraphDataset(root=str(outdir), dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

    print(f"preparing {dataset}_test.pt in pytorch format!")
    test_data = GraphDataset(root=str(outdir), dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


if __name__ == "__main__":
    main()
