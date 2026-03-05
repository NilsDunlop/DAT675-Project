import pandas as pd
import pickle
import torch
import torchani
import torchani_mod
import qcelemental as qcel
import numpy as np
from tqdm import tqdm
from rdkit import Chem
import argparse
from scipy.spatial.distance import cdist

def elements_to_atomicnums(elements):
    atomicnums = np.zeros(len(elements), dtype=int)

    for idx, e in enumerate(elements):
        atomicnums[idx] = qcel.periodictable.to_Z(e)

    return atomicnums


def LoadMolasDF(mol):
    m = mol

    atoms = []

    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)

    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX","ATOM_TYPE","X","Y","Z"]
    
    return df


def LoadPDBasDF(PDB, atom_keys):
    prot_atoms = []
    
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (len(i[12:16].replace(" ","")) < 4 and i[12:16].replace(" ","")[0] != "H") or (len(i[12:16].replace(" ","")) == 4 and i[12:16].replace(" ","")[1] != "H" and i[12:16].replace(" ","")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                         i[17:20]+"-"+i[12:16].replace(" ",""),
                         float(i[30:38]),
                         float(i[38:46]),
                         float(i[46:54])
                        ])
                
    f.close()
    
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX","PDB_ATOM","X","Y","Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(by="ATOM_INDEX").reset_index(drop=True)
    return df


def GetMolAEVs_extended(protein_path, mol, atom_keys, radial_coefs, atom_map):
    # Protein and ligand structure are loaded as pandas DataFrame
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)
    
    # Define AEV coeficients
    # Radial coefficients
    RcR = radial_coefs[0]
    EtaR = radial_coefs[1]
    RsR = radial_coefs[2]
    
    # Angular coefficients (Ga)
    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA = torch.tensor([1.0]) # Angular shift in GA
    EtaA = torch.tensor([1.0])
    RsA = torch.tensor([1.0])  # Radial shift in GA
    
    # Reduce size of Target df to what we need based on radial cutoff RcR
    distance_cutoff = RcR + 0.1
    
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    Target = Target.merge(atom_map, on='ATOM_TYPE', how='left')
    
    # Create tensors of atomic numbers and coordinates of molecule atoms and 
    # protein atoms combined. Encode molecule atoms as hydrogen
    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len)*6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64)
    atomicnums = atomicnums.unsqueeze(0)
    
    coordinates = pd.concat([Ligand[['X','Y','Z']], Target[['X','Y','Z']]])
    coordinates = torch.tensor(coordinates.values)
    coordinates = coordinates.unsqueeze(0)
    
    # Use torchani_mod to calculate AEVs
    atom_symbols = []
    for i in range(1, 23):
        atom_symbols.append(qcel.periodictable.to_symbol(i))
        
    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, 
                                EtaA, Zeta, RsA, TsA, len(atom_symbols))
    
    SC = torchani.SpeciesConverter(atom_symbols)
    sc = SC((atomicnums, coordinates))
    
    aev = AEVC.forward((sc.species, sc.coordinates), mol_len)
    
    # find indices of columns to keep
    # keep all radial terms
    n = len(atom_symbols)
    n_rad_sub = len(EtaR)*len(RsR)
    indices = list(np.arange(n*n_rad_sub))
    
    return Ligand, aev.aevs.squeeze(0)[:mol_len,indices]


def GetMolAEVs_extended_binary(protein_path, mol, atom_keys, radial_coefs, atom_map):
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)
    
    # Extract the radial cutoff distance
    RcR = radial_coefs[0] 

    # Filter the protein DataFrame to a bounding box around the ligand, 
    # padded by the radial cutoff to minimize downstream distance computations
    distance_cutoff = RcR + 0.1
    
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    atom_type_to_nr = dict(zip(atom_map['ATOM_TYPE'], atom_map['ATOM_NR']))
    Target['ATOM_NR'] = Target['ATOM_TYPE'].map(atom_type_to_nr)
    
    n_atom_types = len(atom_map)
    n_ligand_atoms = len(Ligand)
    
    # Get protein atom coordinates and their type indices
    protein_coords = Target[['X','Y','Z']].to_numpy()
    protein_type_indices = Target['ATOM_NR'].to_numpy() - 1
    
    ligand_coords = Ligand[['X','Y','Z']].to_numpy()
    
    if len(protein_coords) > 0 and len(ligand_coords) > 0:
        distances = cdist(ligand_coords, protein_coords)
    else:
        distances = np.zeros((n_ligand_atoms, 0))
    
    # For each ligand atom, compute binary vector over protein atom types
    binary_aevs = np.zeros((n_ligand_atoms, n_atom_types), dtype=np.float32)
    
    valid_mask = ~np.isnan(protein_type_indices)
    distances = distances[:, valid_mask]
    valid_types = protein_type_indices[valid_mask].astype(int)
    
    for i in range(n_ligand_atoms):
        within_cutoff = distances[i] <= RcR
        types_within_cutoff = valid_types[within_cutoff]
        binary_aevs[i, types_within_cutoff] = 1.0
    
    binary_aevs_tensor = torch.tensor(binary_aevs)
    
    return Ligand, binary_aevs_tensor


def GetMolAEVs_extended_distbinned(protein_path, mol, atom_keys, radial_coefs, atom_map, distance_shells=[2.0, 3.5, 5.1]):
    Target = LoadPDBasDF(protein_path, atom_keys)
    Ligand = LoadMolasDF(mol)
    
    # Extract the radial cutoff distance
    RcR = radial_coefs[0] 

    # Filter the protein DataFrame to a bounding box around the ligand, 
    # padded by the radial cutoff to minimize downstream distance computations
    distance_cutoff = RcR + 0.1
    for i in ["X","Y","Z"]:
        Target = Target[Target[i] < float(Ligand[i].max())+distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min())-distance_cutoff]
    
    atom_type_to_nr = dict(zip(atom_map['ATOM_TYPE'], atom_map['ATOM_NR']))
    Target['ATOM_NR'] = Target['ATOM_TYPE'].map(atom_type_to_nr)
    
    n_atom_types = len(atom_map)
    n_ligand_atoms = len(Ligand)
    n_shells = len(distance_shells)
    
    # Get protein atom coordinates and their type indices
    protein_coords = Target[['X','Y','Z']].to_numpy()
    protein_type_indices = Target['ATOM_NR'].to_numpy() - 1
    
    ligand_coords = Ligand[['X','Y','Z']].to_numpy()
    
    if len(protein_coords) > 0 and len(ligand_coords) > 0:
        distances = cdist(ligand_coords, protein_coords)
    else:
        distances = np.zeros((n_ligand_atoms, 0))
        
    valid_mask = ~np.isnan(protein_type_indices)
    distances = distances[:, valid_mask]
    valid_types = protein_type_indices[valid_mask].astype(int)
    
    # For each ligand atom: n_shells counts per atom type
    distbinned_aevs = np.zeros((n_ligand_atoms, n_shells * n_atom_types), dtype=np.float32)
    
    for i in range(n_ligand_atoms):
        dists = distances[i]
        
        shell_start = 0.0
        for shell_idx, shell_end in enumerate(distance_shells):
            in_shell = (dists > shell_start) & (dists <= shell_end)
            types_in_shell = valid_types[in_shell]
            
            if len(types_in_shell) > 0:
                counts = np.bincount(types_in_shell, minlength=n_atom_types)
                distbinned_aevs[i, shell_idx * n_atom_types : (shell_idx + 1) * n_atom_types] = counts[:n_atom_types]
            
            shell_start = shell_end
    
    distbinned_aevs_tensor = torch.tensor(distbinned_aevs)
    
    return Ligand, distbinned_aevs_tensor


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, features=["atom_symbol",
                                  "num_heavy_atoms", 
                                  "total_num_Hs", 
                                  "explicit_valence", 
                                  "is_aromatic", 
                                  "is_in_ring"]):
    # Computes the ligand atom features for graph node construction
    # The standard features are the following:
    # atom_symbol = one hot encoding of atom symbol
    # num_heavy_atoms = # of heavy atom neighbors
    # total_num_Hs = # number of hydrogen atom neighbors
    # explicit_valence = explicit valence of the atom
    # is_aromatic = boolean 1 - aromatic, 0 - not aromatic
    # is_in_ring = boolean 1 - is in ring, 0 - is not in ring

    feature_list = []
    if "atom_symbol" in features:
        feature_list.extend(one_of_k_encoding(atom.GetSymbol(),['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']))
    if "num_heavy_atoms" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))
    if "total_num_Hs" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))
    if "explicit_valence" in features: #-NEW ADDITION FOR PLIG
        feature_list.append(atom.GetExplicitValence())
    if "is_aromatic" in features:
        
        if atom.GetIsAromatic():
            feature_list.append(1)
        else:
            feature_list.append(0)
    if "is_in_ring" in features:
        if atom.IsInRing():
            feature_list.append(1)
        else:
            feature_list.append(0)
    
    return np.array(feature_list)


def mol_to_graph(mol, mol_df, aevs, extra_features=["atom_symbol",
                                                    "num_heavy_atoms", 
                                                    "total_num_Hs", 
                                                    "explicit_valence",
                                                    "is_aromatic",
                                                    "is_in_ring"]):

    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0
    
    # Generate nodes
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H": # Include only non-hydrogen atoms
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            feature = np.append(atom_features(atom), aevs[aev_idx,:])
            features.append(feature)
            counter += 1
    
    #Generate edges
    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            bond_type = one_of_k_encoding(bond.GetBondType(),[1,12,2,3])
            bond_type = [float(b) for b in bond_type]
            edge1 = [idx_to_idx[idx1], idx_to_idx[idx2]]
            edge1.extend(bond_type)
            edge2 = [idx_to_idx[idx2], idx_to_idx[idx1]]
            edge2.extend(bond_type)
            edges.append(edge1)
            edges.append(edge2)
    
    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic', 'double', 'triple'])
    df = df.sort_values(by=['atom1','atom2'])
    
    edge_index = df[['atom1','atom2']].to_numpy().tolist()
    edge_attr = df[['single','aromatic','double','triple']].to_numpy().tolist()
    
    
    return len(mol_df), features, edge_index, edge_attr


"""
Load data
"""
df = pd.read_csv("data/bindingnet_processed.csv", index_col=0)
folder =  "data/bindingnet/from_chembl_client/"

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='original', choices=['binary', 'distance-binned', 'reduced-gaussian-4', 'reduced-gaussian-8', 'original'], help='Encoding scheme to use for AEVs')
args = parser.parse_args()
tag = args.tag
print(f"Using encoding scheme: {tag}")

"""
Generate for all complexes: ANI-2x with 22 atom types. Only 2-atom interactions.
"""
atom_keys = pd.read_csv("data/PDB_Atom_Keys.csv", sep=",")
atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
atom_map[1] = list(np.arange(len(atom_map)) + 1)
atom_map = atom_map.rename(columns={0:"ATOM_TYPE", 1:"ATOM_NR"})

# Radial coefficients: ANI-2x
RcR = 5.1 # Radial cutoff
EtaR = torch.tensor([19.7]) # Radial decay

# ------------------------------------------------------------------------
if tag == 'reduced-gaussian-4':
    RsR = torch.tensor([0.80, 2.14, 3.49, 4.83]) # 4 shifts evenly spaced
elif tag == 'reduced-gaussian-8':
    RsR = torch.tensor([0.80, 1.38, 1.95, 2.53, 3.10, 3.68, 4.25, 4.83]) # 8 shifts
else:
    RsR = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68, 
                    2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83]) # Radial shift
# ------------------------------------------------------------------------

radial_coefs = [RcR, EtaR, RsR]

mol_graphs = {}
for index, row in tqdm(df.iterrows()):
    unique_identify = row['unique_identify']
    target = row['target']
    pdb = row['pdb']
    compnd = row['compnd']

    sdf_file = folder + f"{pdb}/target_{target}/{compnd}/{pdb}_{target}_{compnd}.sdf"
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    lig = suppl[0]

    protein_path = folder + f"{pdb}/rec_h_opt.pdb"

    if tag == 'binary':
        mol_df, aevs = GetMolAEVs_extended_binary(protein_path, lig, atom_keys, radial_coefs, atom_map)
    elif tag == 'distance-binned':
        mol_df, aevs = GetMolAEVs_extended_distbinned(protein_path, lig, atom_keys, radial_coefs, atom_map)
    else:
        mol_df, aevs = GetMolAEVs_extended(protein_path, lig, atom_keys, radial_coefs, atom_map)
    graph = mol_to_graph(lig, mol_df, aevs)
    mol_graphs[unique_identify] = graph

#save the graphs to use as input for the GNN models
suffix = f"_{tag}" if tag != "original" else ""
output_file_graphs = f"data/bindingnet{suffix}.pickle"
with open(output_file_graphs, 'wb') as handle:
    pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)