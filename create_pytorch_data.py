import pandas as pd
import pickle
import argparse
from utils import GraphDataset


def load_benchmark_test_ids():
    """Load all benchmark test set IDs that must be excluded from training."""
    casf = set(pd.read_csv("evaluate/casf-2016/casf2016_test.csv")["unique_id"])
    ligandbias = set(pd.read_csv("evaluate/0ligandbias/0ligandbias_test.csv")["unique_id"])
    oodtest = set(pd.read_csv("evaluate/ood-test/oodtest_test.csv")["unique_id"])
    
    all_test_ids = casf | ligandbias | oodtest
    
    print(f"Benchmark test IDs to exclude from training:")
    print(f"CASF-2016: {len(casf)}")
    print(f"0-LigandBias: {len(ligandbias)}")
    print(f"OOD Test: {len(oodtest)}")
    print(f"Total unique: {len(all_test_ids)}")
    
    return all_test_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_exclusion', action='store_true',
                        help='Skip benchmark exclusion')
    parser.add_argument('--tag', type=str, default='original', choices=['binary', 'distance-binned', 'reduced-gaussian-4', 'reduced-gaussian-8', 'original'], help='Encoding scheme used for AEVs')
    args = parser.parse_args()
    
    tag = args.tag
    suffix = f"_{tag}" if tag != "original" else ""

    """
    Load graphs
    """
    print(f"loading graph from pickle file for pdbbind2020 with tag {tag}")
    with open(f"data/pdbbind{suffix}.pickle", 'rb') as handle:
        pdbbind_graphs = pickle.load(handle)

    print(f"loading graph from pickle file for BindingNet with tag {tag}")
    with open(f"data/bindingnet{suffix}.pickle", 'rb') as handle:
        bindingnet_graphs = pickle.load(handle)

    print(f"loading graph from pickle file for BindingDB with tag {tag}")
    with open(f"data/bindingdb{suffix}.pickle", 'rb') as handle:
        bindingdb_graphs = pickle.load(handle)

    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}


    """
    Load benchmark test IDs for exclusion
    """
    if not args.skip_exclusion:
        exclude_ids = load_benchmark_test_ids()
    else:
        print("WARNING: Skipping benchmark exclusion — training data may leak into test sets!")
        exclude_ids = set()


    """
    Generate data for enriched training for <0.9 Tanimoto to the FEP benchmark
    """
    pdbbind = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
    pdbbind = pdbbind[['PDB_code','-logKd/Ki','split_core','max_tanimoto_fep_benchmark']]
    pdbbind = pdbbind.rename(columns={'PDB_code':'unique_id', 'split_core':'split', '-logKd/Ki':'pK'})
    pdbbind = pdbbind[pdbbind["max_tanimoto_fep_benchmark"] < 0.9]
    pdbbind = pdbbind[['unique_id','pK','split']]

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

    bindingnet = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
    bindingnet = bindingnet.rename(columns={'-logAffi': 'pK','unique_identify':'unique_id'})[['unique_id','pK','max_tanimoto_fep_benchmark']]
    bindingnet['split'] = 'train'
    bindingnet = bindingnet[bindingnet["max_tanimoto_fep_benchmark"] < 0.9]
    bindingnet = bindingnet[['unique_id','pK','split']]

    bindingdb = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
    bindingdb = bindingdb[['unique_id','pK','max_tanimoto_fep_benchmark']]
    bindingdb['split'] = 'train'
    bindingdb = bindingdb[bindingdb["max_tanimoto_fep_benchmark"] < 0.9]
    bindingdb = bindingdb[['unique_id','pK','split']]

    # combine pdbbind2020, bindingnet, and bindingdb index sets
    data = pd.concat([pdbbind, bindingnet, bindingdb], ignore_index=True)
    print(f"\nFinal split counts:")
    print(data[['split']].value_counts())

    dataset = f'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark{suffix}'

    df = data[data['split'] == 'train']
    train_ids, train_y = list(df['unique_id']), list(df['pK'])

    df = data[data['split'] == 'valid']
    valid_ids, valid_y = list(df['unique_id']), list(df['pK'])

    df = data[data['split'] == 'test']
    test_ids, test_y = list(df['unique_id']), list(df['pK'])

    # make data PyTorch Geometric ready
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = GraphDataset(root='data', dataset=dataset + '_train', ids=train_ids, y=train_y, graphs_dict=graphs_dict)

    print('preparing ', dataset + '_valid.pt in pytorch format!')
    valid_data = GraphDataset(root='data', dataset=dataset + '_valid', ids=valid_ids, y=valid_y, graphs_dict=graphs_dict)

    print('preparing ', dataset + '_test.pt in pytorch format!')
    test_data = GraphDataset(root='data', dataset=dataset + '_test', ids=test_ids, y=test_y, graphs_dict=graphs_dict)


if __name__ == "__main__":
    main()