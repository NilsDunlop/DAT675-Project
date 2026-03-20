from rdkit import Chem
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import itertools
import numpy as np
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

def load_simple_dataset(csv_path, base_dir, source_name):
    df = pd.read_csv(csv_path)
    rows = []

    for _, row in df.iterrows():
        try:
            pdb_id = str(row["unique_id"]).lower()

            sdf_file = row.get("sdf_file")
            mol2_file = row.get("mol2_file")

            path = None

            # Prefer SDF
            if pd.notna(sdf_file):
                sdf_path = os.path.join(base_dir, sdf_file)
                if os.path.exists(sdf_path):
                    path = sdf_path

            # Fallback to MOL2
            if path is None and pd.notna(mol2_file):
                mol2_path = os.path.join(base_dir, mol2_file)
                if os.path.exists(mol2_path):
                    path = mol2_path

            if path is None:
                continue

            smiles = file_to_smiles(path)
            if smiles is None:
                continue

            rows.append({
                "pdb_id": pdb_id,
                "smiles": smiles,
                "pK": row["pK"],
                "source": source_name
            })

        except Exception as e:
            print(f"{source_name} error:", e)
            continue

    return pd.DataFrame(rows, columns=["pdb_id", "smiles", "pK", "source"])

def file_to_smiles(path):
    """
    Load a molecule from MOL2 or SDF, sanitize, remove salts, add Hs,
    kekulize, and return canonical SMILES.
    """
    if not os.path.exists(path):
        return None

    # Load molecule
    mol = None
    if path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(path, sanitize=False)
    elif path.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(path, sanitize=False)
        if len(supplier) > 0:
            mol = supplier[0]
    else:
        return None

    if mol is None:
        return None

    # Sanitize and add hydrogens
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)  # balance formal charges

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)    # Remove salts / keep largest fragment
        if len(frags) > 1:
            # keep the largest fragment (by atom count)
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        
        uncharger = rdMolStandardize.Uncharger() # create a standardizer
        mol = uncharger.uncharge(mol)   # apply to molecule
    except:
        return None

    # Kekulize
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except:
        pass  # sometimes fails on weird aromatic systems, safe to ignore

    # Return canonical SMILES
    return Chem.MolToSmiles(mol, canonical=True)


# ADD BINDINGNET
bindingnet = pd.read_csv("bindingnet_processed.csv")

bindingnet_rows = []

for _, row in bindingnet.iterrows():
    try:
        parts = row["unique_identify"].split("_")
        chembl_protein = parts[0]
        pdb_id = parts[1].lower()
        chembl_ligand = parts[2]

        sdf_path = os.path.join(
            "bindingnet/from_chembl_client",
            pdb_id,
            f"target_{chembl_protein}",
            chembl_ligand,
            f"{pdb_id}_{chembl_protein}_{chembl_ligand}.sdf"
        )
        smiles = file_to_smiles(sdf_path)
        if smiles is None:
            continue

        bindingnet_rows.append({
            "pdb_id": pdb_id,
            "smiles": smiles,
            "pK": row["-logAffi"],
            "source": "bindingnet"
        })
    except Exception as e:
        print("Bindingnet error:", e)
        continue

df_bindingnet = pd.DataFrame(bindingnet_rows, columns=["pdb_id", "smiles", "pK", "source"])
print(len(df_bindingnet))

# ADD BINDINGDB
bindingdb = pd.read_csv("bindingdb_processed.csv")

bindingdb_rows = []

for _, row in bindingdb.iterrows():
    try:
        folder = row["folder"]
        pdb_id = folder.split("_")[0].lower()

        mol2_path = os.path.join(
            "bindingdb/surflex",
            folder,
            row["mol2_file"]
        )

        smiles = file_to_smiles(mol2_path)
        if smiles is None:
            continue

        bindingdb_rows.append({
            "pdb_id": pdb_id,
            "smiles": smiles,
            "pK": row["pK"],
            "source": "bindingdb"
        })
    except Exception as e:
        print("BindingDB error:", e)
        continue

df_bindingdb = pd.DataFrame(bindingdb_rows, columns=["pdb_id", "smiles", "pK", "source"])

# ADD PDBBIND
pdbbind = pd.read_csv("pdbbind_processed.csv")

pdbbind_rows = []

for _, row in pdbbind.iterrows():
    try:
        pdb_id = row["PDB_code"].lower()

        # Try refined first, then general
        mol2_path_refined = os.path.join(
            "pdbbind/refined-set",
            pdb_id,
            f"{pdb_id}_ligand.mol2"
        )

        mol2_path_general = os.path.join(
            "pdbbind/general-set",
            pdb_id,
            f"{pdb_id}_ligand.mol2"
        )

        mol2_path = (
            mol2_path_refined
            if os.path.exists(mol2_path_refined)
            else mol2_path_general
        )

        smiles = file_to_smiles(mol2_path)
        if smiles is None:
            continue

        pdbbind_rows.append({
            "pdb_id": pdb_id,
            "smiles": smiles,
            "pK": row["-logKd/Ki"],
            "source": "pdbbind",
            "split": row["split_core"]
        })
    except Exception as e:
        print("PDBBind error:", e)
        continue

df_pdbbind = pd.DataFrame(pdbbind_rows)

# ADD EVALUATION SETS

df_casf2016 = load_simple_dataset(
    "../evaluate/casf-2016/casf2016_test.csv",
    "../",
    "casf2016"
)

df_0ligandbias = load_simple_dataset(
    "../evaluate/0ligandbias/0ligandbias_test.csv",
    "../",
    "0ligandbias"
)

df_oodtest = load_simple_dataset(
    "../evaluate/ood-test/oodtest_test.csv",
    "../",
    "oodtest"
)

df_fep = load_simple_dataset(
    "fep_streamlined.csv",
    "fep/",
    "fep"
)
# get pK values
df_fep["pK"] = - np.log(10) * df_fep["pK"] * 0.001987 * 297 #


# 1. Properly define splits within dataframes
df_bindingnet["split"] = "train"
df_bindingdb["split"] = "train"
# For PDBbind, check the 'split' column we loaded earlier
df_pdbbind["split"] = df_pdbbind["split"].apply(lambda x: "test" if x == "core" else "train")

df_casf2016["split"] = "test"
df_0ligandbias["split"] = "test"
df_oodtest["split"] = "test"
df_fep["split"] = "test"

# 2. Create Pair Strings helper
def get_pairs(df):
    # Ensure no NaNs and create unique ID
    valid = df.dropna(subset=["pdb_id", "smiles"])
    return set(valid["pdb_id"].str.lower() + "_" + valid["smiles"])

# 3. Generate Sets
pairs_bindingdb = get_pairs(df_bindingdb)
pairs_bindingnet = get_pairs(df_bindingnet)
# ONLY include PDBbind rows marked as train
pairs_pdbbind_train = get_pairs(df_pdbbind[df_pdbbind["split"] == "train"])

# Combine all training pairs
training_pair_set = set.union(pairs_bindingdb, pairs_bindingnet, pairs_pdbbind_train)

# 4. Define Test Sets to check
test_datasets = {
    "CASF-2016": get_pairs(df_casf2016),
    "0-Ligand-Bias": get_pairs(df_0ligandbias),
    "OOD-Test": get_pairs(df_oodtest),
    "FEP-Set": get_pairs(df_fep)
}

print(f"Total unique training pairs: {len(training_pair_set)}")

# 5. Compute Overlap
print("\n--- Overlap with Training Data ---")
for name, test_set in test_datasets.items():
    overlap = len(test_set & training_pair_set)
    percent = (overlap / len(test_set)) * 100 if len(test_set) > 0 else 0
    print(f"{name}: {overlap} / {len(test_set)} ({percent:.2f}%)")

# 6. Pairwise overlap among training sources (Validation)
print("\n--- Training Source Overlap ---")
print("BindingDB ∩ BindingNet:", len(pairs_bindingdb & pairs_bindingnet))
print("BindingDB ∩ PDBBind-Train:", len(pairs_bindingdb & pairs_pdbbind_train))

