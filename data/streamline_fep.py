import os
import pandas as pd

# Load your main dataset
df = pd.read_csv("../evaluate/fep/fep_benchmark.csv")


# Function to extract protein identifier
def extract_protein_id(row):
    group = row["group_abbreviation"]
    full_name = row["pdb_file"]
    
    prefix = group + "_"
    
    if full_name.startswith(prefix):
        name = full_name[len(prefix):]
    else:
        name = full_name  # fallback if something is inconsistent
    
    # Remove .pdb extension
    if name.endswith("_protein.pdb"):
        name = name[:-12]
    
    return name

# Apply extraction
df["protein_id"] = df.apply(extract_protein_id, axis=1)

# Dictionary to store metadata per group (avoid reloading repeatedly)
metadata_cache = {}

# Function to get PDB code
def get_pdb_code(row):
    group = row["group_abbreviation"] 
    protein_id = row["protein_id"]
    
    # Load metadata if not already cached
    if group not in metadata_cache:
        path = os.path.join("../../public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/", group, "subset_metadata.csv")
        if os.path.exists(path):
            metadata_cache[group] = pd.read_csv(path)
        else:
            metadata_cache[group] = None
    
    metadata = metadata_cache[group]
#    print(group)    
#    print(protein_id)    

    if metadata is None:
        print("NO METADATA")
        exit()
        return None
    
    # Adjust column names depending on actual file structure
    # Example assumes columns: "protein_id" and "pdb_code"
    match = metadata[metadata["Input file naming scheme"] == protein_id]
    
    if not match.empty:
        pdb_code = match.iloc[0]["Reference PDB"]
        return str(pdb_code).lower()  # convert to lowercase
    
    print("COULD NOT MATCH META DATA")
    exit()
    return None

# Apply lookup
df["unique_id"] = df.apply(get_pdb_code, axis=1)

df["sdf_file"] = df.apply(
    lambda row: os.path.join(row["group_abbreviation"], row["sdf_file"]),
    axis=1
)
df["pdb_file"] = df.apply(
    lambda row: os.path.join(row["group_abbreviation"], row["pdb_file"]),
    axis=1
)
#df["sdf_file"]  = os.path.join(df["group_abbreviation"], df["sdf_file"] )
#df["pdb_file"]  = os.path.join(df["group_abbreviation"], df["pdb_file"] )

# Save result
df.to_csv("fep_streamlined.csv", index=False)
