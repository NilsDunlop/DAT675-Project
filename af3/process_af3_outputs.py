#!/usr/bin/env python3
"""
process_af3_outputs.py — Convert AF3 CIF outputs to PDB + ligand SDF for AEV-PLIG.

Produces per-entry directories and an evaluation CSV compatible with AEV-PLIG's
process_and_predict.py pipeline.

Output structure:
    {output_dir}/{benchmark}/{pdb_id}/
        {pdb_id}_protein.pdb     — protein chains only (gemmi)
        {pdb_id}_ligand.sdf      — ligand with bond orders from CCD template (RDKit)
    {output_dir}/{benchmark}_af3_test.csv  — AEV-PLIG evaluation CSV

Dependencies:
    pip install gemmi rdkit-pypi

Usage:
    python process_af3_outputs.py \
        --af3_dir af_output \
        --output_dir af_processed \
        --verification_csv output/af3_verification.csv

    # Then evaluate with AEV-PLIG:
    python process_and_predict.py \
        --dataset_csv=af_processed/casf2016_af3_test.csv \
        --data_name=af3_casf2016 \
        --trained_model_name=... --device=0
"""

import argparse
import csv
import json
import logging
import os
import sys
import io

try:
    import gemmi
except ImportError:
    sys.exit("ERROR: gemmi not installed. Run: pip install gemmi")

try:
    from rdkit import Chem, Geometry
    from rdkit.Chem import AllChem, rdMolDescriptors, rdmolops
except ImportError:
    sys.exit("ERROR: RDKit not installed. Run: pip install rdkit-pypi\n"
             "RDKit is required for SDF extraction with correct bond orders.")


# ── Special-case entries (no small-molecule ligand) ──────────────────────
PEPTIDE_LIGAND_ENTRIES = {
    "1a30", "3bv9", "3uri",           # CASF-2016 peptide ligands
    "1jrs", "1tl9", "1pop",           # 0LigandBias leupeptin
    "2l3r", "2l75", "4yhp",           # 0LigandBias H3K9me3 peptides
}
PPI_ENTRIES = {"1g1e", "1s5q"}
NO_SDF_ENTRIES = PEPTIDE_LIGAND_ENTRIES | PPI_ENTRIES


def extract_protein_pdb(cif_path, protein_pdb_path):
    """
    Extract protein-only PDB from CIF using gemmi.
    Removes waters, ligands, and non-polymer entities.
    Keeps only polypeptide chains (ATOM records).
    """
    st = gemmi.read_structure(cif_path)
    st.setup_entities()

    # Remove waters and ligands
    st.remove_waters()
    st.remove_ligands_and_waters()

    st.write_pdb(protein_pdb_path)
    return True


def extract_ligand_sdf(cif_path, sdf_path, pdb_id, ccd_code):
    """
    Extract ligand from AF3 CIF and write SDF with correct bond orders.

    Strategy:
    1. Parse CIF with gemmi → get ligand HETATM coords + atom names
    2. Build RDKit mol from CCD SMILES or template → correct bond orders
    3. Assign AF3-predicted 3D coordinates by atom name matching
    4. Write SDF

    This ensures AEV-PLIG gets proper bond types (single/double/aromatic/triple)
    which are required for its graph construction and edge features.
    """
    st = gemmi.read_structure(cif_path)
    if not st or not st[0]:
        return False, "empty structure"

    model = st[0]

    # Find ligand residue matching CCD code
    ligand_res = None
    for chain in model:
        for residue in chain:
            if residue.name == ccd_code:
                ligand_res = residue
                break
        if ligand_res:
            break

    if not ligand_res:
        return False, f"ligand {ccd_code} not found in CIF"

    # Collect predicted coordinates indexed by atom name, with element from gemmi
    pred_coords = {}
    pred_elements = {}
    for atom in ligand_res:
        name = atom.name.strip()
        pred_coords[name] = (atom.pos.x, atom.pos.y, atom.pos.z)
        pred_elements[name] = atom.element.name  # gemmi knows the correct element

    if len(pred_coords) == 0:
        return False, "no atoms in ligand residue"

    # ── Approach 1: Build mol from CCD ideal SDF via gemmi monomer lib ───
    # Try to get the CCD component from gemmi's built-in chemical components
    mol_with_bonds = None
    try:
        mol_with_bonds = _build_mol_from_ccd_gemmi(ccd_code, pred_coords, pred_elements)
    except Exception as e:
        logging.debug(f"  {pdb_id}: gemmi CCD approach failed: {e}")

    # ── Approach 2: Build PDB block → RDKit → assign bonds via template ──
    if mol_with_bonds is None:
        try:
            mol_with_bonds = _build_mol_from_pdb_block(
                ligand_res, ccd_code, pred_coords, pdb_id, pred_elements
            )
        except Exception as e:
            logging.debug(f"  {pdb_id}: PDB block approach failed: {e}")

    if mol_with_bonds is None:
        return False, "could not build mol with bond orders"

    # Verify we actually have non-trivial bond orders before writing
    heavy_mol = Chem.RemoveHs(mol_with_bonds, sanitize=False)
    bond_types_pre = set(b.GetBondTypeAsDouble() for b in heavy_mol.GetBonds())

    # Kekulize: converts aromatic bonds (1.5) to explicit alternating
    # single/double for V2000 SDF format
    kekulized = False
    try:
        # Work on a copy so we don't corrupt the original if kekulize fails
        mol_to_write = Chem.RWMol(mol_with_bonds)
        Chem.SanitizeMol(mol_to_write)
        Chem.Kekulize(mol_to_write, clearAromaticFlags=True)
        kekulized = True
    except Exception:
        try:
            mol_to_write = Chem.RWMol(mol_with_bonds)
            Chem.Kekulize(mol_to_write, clearAromaticFlags=True)
            kekulized = True
        except Exception as e:
            logging.debug(f"  {pdb_id}: Kekulization failed: {e}")
            mol_to_write = mol_with_bonds

    # Write SDF
    try:
        writer = Chem.SDWriter(sdf_path)
        writer.SetForceV3000(False)
        if kekulized:
            writer.SetKekulize(False)  # Already kekulized manually
        else:
            writer.SetKekulize(True)   # Let SDWriter try
        writer.write(mol_to_write)
        writer.close()
        return True, ""
    except Exception as e:
        return False, f"SDF write failed: {e}"


# ── Global CCD component cache ───────────────────────────────────────────
_CCD_DOC = None
_CCD_SMILES_CACHE = {}


def _load_ccd_components():
    """Load the monolithic components.cif(.gz) once into a gemmi Document."""
    global _CCD_DOC
    if _CCD_DOC is not None:
        return _CCD_DOC

    ccd_dir = os.environ.get("CLIBD_MON", "")
    if not ccd_dir:
        return None

    # Try both compressed and uncompressed
    for fname in ["components.cif.gz", "components.cif"]:
        path = os.path.join(ccd_dir, fname)
        if os.path.exists(path):
            logging.info(f"Loading CCD components from {path} (this may take ~30s)...")
            _CCD_DOC = gemmi.cif.read(path)
            logging.info(f"CCD loaded: {len(_CCD_DOC)} component blocks")
            return _CCD_DOC

    # Also check CCP4-style layout: {ccd_dir}/{first_letter}/{CCD}.cif
    # In this case we don't preload but return a sentinel
    if os.path.isdir(os.path.join(ccd_dir, "a")):
        _CCD_DOC = "CCP4_LAYOUT"
        return _CCD_DOC

    return None


def _get_ccd_smiles(ccd_code):
    """
    Look up the canonical SMILES for a CCD code from the components dictionary.
    Caches results for speed.
    """
    if ccd_code in _CCD_SMILES_CACHE:
        return _CCD_SMILES_CACHE[ccd_code]

    doc = _load_ccd_components()
    if doc is None:
        return None

    block = None
    if doc == "CCP4_LAYOUT":
        # CCP4-style: individual CIF files per component
        ccd_dir = os.environ.get("CLIBD_MON", "")
        cif_path = os.path.join(ccd_dir, ccd_code[0].lower(), f"{ccd_code}.cif")
        if os.path.exists(cif_path):
            try:
                block = gemmi.cif.read(cif_path)[0]
            except Exception:
                pass
    else:
        # Monolithic components.cif — find the block by name
        block = doc.find_block(ccd_code)

    if block is None:
        _CCD_SMILES_CACHE[ccd_code] = None
        return None

    # Extract SMILES from _pdbx_chem_comp_descriptor
    smiles = None
    try:
        table = block.find(["_pdbx_chem_comp_descriptor.descriptor",
                            "_pdbx_chem_comp_descriptor.type"])
        for row in table:
            desc_type = row[1].upper() if row[1] else ""
            if "SMILES_CANONICAL" in desc_type and "OPENEYE" in desc_type:
                smiles = row[0]
                break
            if "SMILES_CANONICAL" in desc_type and smiles is None:
                smiles = row[0]
            if "SMILES" in desc_type and smiles is None:
                smiles = row[0]
    except Exception:
        pass

    # CIF format wraps string values in quotes — strip them for RDKit
    if smiles:
        smiles = smiles.strip().strip('"').strip("'")

    _CCD_SMILES_CACHE[ccd_code] = smiles
    return smiles


def _build_mol_from_ccd_gemmi(ccd_code, pred_coords, pred_elements=None):
    """
    Primary approach: use pdbeccdutils to load the CCD component (correct
    bond orders), then read atom names from the CCD CIF block via gemmi
    (since pdbeccdutils doesn't populate PDBResidueInfo), and assign AF3
    predicted coordinates by matching CCD atom names to CIF atom names.
    """
    try:
        from pdbeccdutils.core import ccd_reader
    except ImportError:
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    # Get CCD block from cache
    global _CCD_DOC
    if _CCD_DOC is None:
        _load_ccd_components()
    if _CCD_DOC is None or _CCD_DOC == "CCP4_LAYOUT":
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    block = _CCD_DOC.find_block(ccd_code)
    if block is None:
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    # ── Step 1: Read CCD atom names and their order from gemmi ──
    # The _chem_comp_atom table has the canonical atom names in order
    ccd_atom_names = []  # ordered list of heavy atom names from CCD
    try:
        atom_table = block.find(["_chem_comp_atom.atom_id",
                                  "_chem_comp_atom.type_symbol"])
        for row in atom_table:
            atom_name = row[0].strip().strip('"').strip("'")
            element = row[1].strip().strip('"').strip("'")
            if element != "H":
                ccd_atom_names.append(atom_name)
    except Exception as e:
        logging.debug(f"  {ccd_code}: Failed to read _chem_comp_atom: {e}")
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    if not ccd_atom_names:
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    # ── Step 2: Get pdbeccdutils mol with correct bond orders ──
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp:
        tmp.write(block.as_string())
        tmp_path = tmp.name

    try:
        result = ccd_reader.read_pdb_cif_file(tmp_path)
    except Exception as e:
        logging.debug(f"  {ccd_code}: pdbeccdutils read failed: {e}")
        os.unlink(tmp_path)
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    component = result.component
    mol = component.mol
    if mol is None or mol.GetNumAtoms() == 0:
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    # Remove H from mol
    mol = Chem.RemoveHs(mol)

    try:
        conf = mol.GetConformer()
    except Exception:
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    # ── Step 3: Map CCD atom names to RDKit atom indices ──
    # pdbeccdutils preserves the atom order from the CCD CIF, so
    # the i-th heavy atom in mol corresponds to ccd_atom_names[i]
    n_mol_heavy = mol.GetNumAtoms()  # all heavy after RemoveHs
    if n_mol_heavy != len(ccd_atom_names):
        logging.debug(f"  {ccd_code}: atom count mismatch mol={n_mol_heavy} vs CCD={len(ccd_atom_names)}")
        # Try anyway with the minimum
        n_map = min(n_mol_heavy, len(ccd_atom_names))
    else:
        n_map = n_mol_heavy

    ccd_name_to_idx = {}
    for i in range(n_map):
        ccd_name_to_idx[ccd_atom_names[i]] = i

    # ── Step 4: Assign AF3 predicted coordinates by atom name ──
    cif_heavy = {}
    for name, (x, y, z) in pred_coords.items():
        elem = pred_elements.get(name, "C") if pred_elements else "C"
        if elem != "H":
            cif_heavy[name] = (x, y, z)

    assigned = 0
    for cif_name, (x, y, z) in cif_heavy.items():
        if cif_name in ccd_name_to_idx:
            idx = ccd_name_to_idx[cif_name]
            conf.SetAtomPosition(idx, Geometry.Point3D(x, y, z))
            assigned += 1

    if assigned == 0:
        logging.debug(f"  {ccd_code}: No atom name matches. CIF: {list(cif_heavy.keys())[:5]}, CCD: {ccd_atom_names[:5]}")
        return _build_mol_from_smiles(ccd_code, pred_coords, pred_elements)

    n_heavy_cif = len(cif_heavy)
    if assigned < n_heavy_cif * 0.8:
        logging.warning(f"  {ccd_code}: Only {assigned}/{n_heavy_cif} atoms matched by name")

    # Add H back
    mol = Chem.AddHs(mol, addCoords=True)

    return mol


def _build_mol_from_smiles(ccd_code, pred_coords, pred_elements=None):
    """
    Fallback: build mol from CCD SMILES (correct bond orders),
    then assign AF3 predicted 3D coordinates by element matching.
    Used when pdbeccdutils is not available.
    """
    smiles = _get_ccd_smiles(ccd_code)
    if not smiles:
        return None

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        return None

    # Add H to template to get full molecule
    mol = Chem.AddHs(template)

    # Generate initial 3D coords (needed as starting point)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    except Exception:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception:
            return None

    conf = mol.GetConformer()

    # Separate CIF coords into heavy atoms by element
    cif_heavy = {}
    for name, (x, y, z) in pred_coords.items():
        elem = pred_elements.get(name, "C") if pred_elements else "C"
        if elem != "H":
            cif_heavy.setdefault(elem, []).append((name, x, y, z))

    # Collect template heavy atoms by element
    tmpl_heavy = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            elem = atom.GetSymbol()
            tmpl_heavy.setdefault(elem, []).append(atom.GetIdx())

    assigned = 0
    for elem, atom_idxs in tmpl_heavy.items():
        cif_atoms = cif_heavy.get(elem, [])
        if not cif_atoms:
            continue
        for idx, (name, x, y, z) in zip(atom_idxs, cif_atoms[:len(atom_idxs)]):
            conf.SetAtomPosition(idx, Geometry.Point3D(x, y, z))
            assigned += 1

    if assigned == 0:
        return None

    return mol


def _build_mol_from_pdb_block(ligand_res, ccd_code, pred_coords, pdb_id, pred_elements=None):
    """
    Fallback: build ligand mol from PDB HETATM block with proximity bonding,
    then try to assign bond orders from CCD SMILES template.
    Only used when _build_mol_from_ccd_gemmi fails (no SMILES, etc.)
    """
    pdb_block = _make_pdb_block(ccd_code, pred_coords, pred_elements)

    # RDKit can read PDB blocks and infer connectivity from distances
    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False,
                                proximityBonding=True)
    if mol is None:
        return None

    # Strip H before template matching
    mol_noH = Chem.RemoveHs(mol, sanitize=False)

    # Try to determine bond orders from the molecular graph
    try:
        smiles = _get_ccd_smiles(ccd_code)
        if smiles:
            template = Chem.MolFromSmiles(smiles)
            if template is not None:
                template_noH = Chem.RemoveHs(template)
                mol_noH = AllChem.AssignBondOrdersFromTemplate(template_noH, mol_noH)
    except Exception as e:
        logging.debug(f"  {pdb_id}: Template bond assignment failed: {e}")
        # Fall through - we still have the mol with proximity bonds

    try:
        Chem.SanitizeMol(mol_noH)
    except Exception:
        # Try partial sanitization
        try:
            Chem.SanitizeMol(mol_noH, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                            Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        except Exception:
            pass

    # Add hydrogens back
    mol_withH = Chem.AddHs(mol_noH, addCoords=True)

    # Verify we have non-trivial bond orders
    bond_types = set()
    for bond in mol_withH.GetBonds():
        bond_types.add(bond.GetBondTypeAsDouble())
    if bond_types == {1.0} and mol_withH.GetNumHeavyAtoms() > 3:
        logging.warning(f"  {pdb_id}: All bonds are single - bond orders "
                       f"may not have been assigned correctly for {ccd_code}")

    return mol_withH


def _make_pdb_block(ccd_code, pred_coords, pred_elements=None):
    """Build a PDB-format text block from ligand coordinates.
    
    Args:
        ccd_code: CCD identifier
        pred_coords: dict of atom_name -> (x, y, z)
        pred_elements: dict of atom_name -> element symbol (from gemmi).
                       If None, element is guessed from atom name (unreliable).
    """
    lines = [f"HEADER    {ccd_code}"]
    for i, (name, (x, y, z)) in enumerate(pred_coords.items(), 1):
        # Use gemmi-provided element if available
        if pred_elements and name in pred_elements:
            element = pred_elements[name]
        else:
            # Fallback: guess from atom name (unreliable for Cl, Br, Fe, etc.)
            element = name.strip()
            if len(element) > 1:
                element = ''.join(c for c in element if not c.isdigit())
            if len(element) > 2:
                element = element[:2]
            if len(element) == 2 and element[1].isupper():
                element = element[0]

        # Format atom name per PDB convention
        # 2-char elements (Cl, Br, Fe) start at column 13; 1-char at column 14
        if len(element) == 2:
            atom_name = f"{name:<4s}"
        elif len(name) < 4:
            atom_name = f" {name:<3s}"
        else:
            atom_name = f"{name:<4s}"

        line = (
            f"HETATM{i:5d} {atom_name:4s} {ccd_code:>3s} A"
            f"   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          "
            f"{element:>2s}"
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


def get_ccd_code_from_data_json(data_json_path):
    """Extract the CCD code from the AF3 _data.json file."""
    try:
        with open(data_json_path) as f:
            data = json.load(f)
        for seq in data.get("sequences", []):
            if "ligand" in seq:
                codes = seq["ligand"].get("ccdCodes", [])
                if codes:
                    return codes[0]
    except Exception:
        pass
    return None


def get_pk_from_verification_csv(csv_path):
    """Load pK values and CCD codes from the af3_verification.csv."""
    pk_map = {}
    ccd_map = {}
    if not csv_path or not os.path.exists(csv_path):
        return pk_map, ccd_map
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get("pdb_id", "").strip().lower()
            pk = row.get("pk", "").strip()
            ccd = row.get("ccd_code_used", "").strip()
            if pdb_id:
                if pk:
                    pk_map[pdb_id] = pk
                if ccd and ccd != "PEPTIDE":
                    ccd_map[pdb_id] = ccd
    return pk_map, ccd_map


def process_entry(benchmark, pdb_id, af3_dir, output_dir, ccd_code=None):
    """Process a single AF3 output entry."""
    job_name = f"{pdb_id}_{benchmark}"
    entry_dir = os.path.join(af3_dir, benchmark, job_name)

    model_cif = os.path.join(entry_dir, f"{job_name}_model.cif")
    data_json = os.path.join(entry_dir, f"{job_name}_data.json")

    if not os.path.exists(model_cif):
        return {"status": "missing", "reason": "model.cif not found"}

    out_dir = os.path.join(output_dir, benchmark, pdb_id)
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "pdb_id": pdb_id,
        "benchmark": benchmark,
        "status": "ok",
        "ccd_code": ccd_code or "",
        "has_protein_pdb": False,
        "has_ligand_sdf": False,
        "protein_pdb_path": "",
        "ligand_sdf_path": "",
        "ligand_note": "",
    }

    # 1. Protein-only PDB
    protein_pdb = os.path.join(out_dir, f"{pdb_id}_protein.pdb")
    try:
        extract_protein_pdb(model_cif, protein_pdb)
        result["has_protein_pdb"] = True
        result["protein_pdb_path"] = protein_pdb
    except Exception as e:
        logging.error(f"  {pdb_id}: Protein PDB extraction failed: {e}")
        result["status"] = "error"
        result["reason"] = str(e)
        return result

    # 2. Ligand SDF
    if pdb_id.lower() in NO_SDF_ENTRIES:
        result["has_ligand_sdf"] = False
        result["ligand_note"] = "peptide/PPI entry"
    else:
        if not ccd_code and os.path.exists(data_json):
            ccd_code = get_ccd_code_from_data_json(data_json)
            result["ccd_code"] = ccd_code or ""

        if ccd_code:
            ligand_sdf = os.path.join(out_dir, f"{pdb_id}_ligand.sdf")
            ok, note = extract_ligand_sdf(model_cif, ligand_sdf, pdb_id, ccd_code)
            result["has_ligand_sdf"] = ok
            result["ligand_sdf_path"] = ligand_sdf if ok else ""
            result["ligand_note"] = note

            if ok:
                # Validate: check RDKit can read it back with bonds
                try:
                    suppl = Chem.SDMolSupplier(ligand_sdf, removeHs=False)
                    mol = next(iter(suppl))
                    if mol is None:
                        result["ligand_note"] = "SDF written but RDKit cannot re-read"
                        result["has_ligand_sdf"] = False
                    else:
                        n_atoms = mol.GetNumAtoms()
                        n_bonds = mol.GetNumBonds()
                        bond_types = set(b.GetBondTypeAsDouble() for b in mol.GetBonds())
                        result["ligand_note"] = (
                            f"{n_atoms} atoms, {n_bonds} bonds, "
                            f"types={sorted(bond_types)}"
                        )
                except Exception as e:
                    result["ligand_note"] = f"validation failed: {e}"
        else:
            result["ligand_note"] = "no CCD code"

    return result


def write_evaluation_csv(results, pk_map, output_dir, benchmark):
    """
    Write AEV-PLIG compatible evaluation CSV.

    Format: unique_id,pK,sdf_file,mol2_file,pdb_file
    (mol2_file left empty since AF3 outputs don't have mol2;
     AEV-PLIG will use SDF path when --use_mol2 is not passed)
    """
    csv_path = os.path.join(output_dir, f"{benchmark}_af3_test.csv")

    rows = []
    for r in results:
        if r["benchmark"] != benchmark:
            continue
        if not r["has_protein_pdb"] or not r["has_ligand_sdf"]:
            continue

        pdb_id = r["pdb_id"]
        rows.append({
            "unique_id": pdb_id,
            "pK": pk_map.get(pdb_id.lower(), ""),
            "sdf_file": r["ligand_sdf_path"],
            "mol2_file": "",  # Not available from AF3
            "pdb_file": r["protein_pdb_path"],
        })

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "unique_id", "pK", "sdf_file", "mol2_file", "pdb_file"
        ])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert AF3 CIF outputs to PDB + SDF for AEV-PLIG"
    )
    parser.add_argument("--af3_dir", default="af_output",
                        help="Root AF3 output directory")
    parser.add_argument("--output_dir", default="af_processed",
                        help="Output directory for processed files")
    parser.add_argument("--verification_csv", default="",
                        help="Path to af3_verification.csv (for CCD codes)")
    parser.add_argument("--benchmark_csvs", nargs="*", default=[],
                        help="Original benchmark CSVs with pK values "
                             "(e.g. casf2016_test.csv 0ligandbias_test.csv)")
    parser.add_argument("--benchmark", default="",
                        help="Process only this benchmark")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info(f"gemmi version: {gemmi.__version__}")
    logging.info(f"RDKit version: {Chem.rdBase.rdkitVersion}")

    # Check for CCD monomer library
    monlib = os.environ.get("CLIBD_MON", "")
    if monlib:
        logging.info(f"CCP4 monomer library: {monlib}")
    else:
        logging.info("CLIBD_MON not set — will use RDKit proximity bonding + template matching")

    # Load CCD codes from verification CSV
    pk_map, ccd_map = get_pk_from_verification_csv(args.verification_csv)
    if ccd_map:
        logging.info(f"Loaded {len(ccd_map)} CCD codes from {args.verification_csv}")

    # Load pK values from original benchmark CSVs
    for bench_csv in args.benchmark_csvs:
        if not os.path.exists(bench_csv):
            logging.warning(f"Benchmark CSV not found: {bench_csv}")
            continue
        with open(bench_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("unique_id", "").strip().lower()
                pk = row.get("pK", row.get("pk", "")).strip()
                if uid and pk:
                    pk_map[uid] = pk
        logging.info(f"Loaded pK values from {bench_csv} (total: {len(pk_map)})")

    if pk_map:
        logging.info(f"Total pK values: {len(pk_map)}")

    # Discover benchmarks
    if args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = sorted([
            d for d in os.listdir(args.af3_dir)
            if os.path.isdir(os.path.join(args.af3_dir, d))
        ])

    all_results = []
    totals = {"ok": 0, "error": 0, "missing": 0}

    for benchmark in benchmarks:
        bench_dir = os.path.join(args.af3_dir, benchmark)
        entries = sorted([
            d for d in os.listdir(bench_dir)
            if os.path.isdir(os.path.join(bench_dir, d))
            and not d.startswith(".")
        ])

        logging.info(f"\n{'='*60}")
        logging.info(f"{benchmark}: {len(entries)} entries")
        logging.info(f"{'='*60}")

        bench_results = []
        for job_name in entries:
            parts = job_name.rsplit(f"_{benchmark}", 1)
            pdb_id = parts[0] if parts else job_name

            ccd_code = ccd_map.get(pdb_id.lower())

            result = process_entry(
                benchmark, pdb_id, args.af3_dir, args.output_dir, ccd_code
            )
            result["benchmark"] = benchmark
            bench_results.append(result)
            all_results.append(result)

            status = result["status"]
            totals[status] = totals.get(status, 0) + 1

            if status != "ok":
                logging.warning(f"  {pdb_id}: {status} - {result.get('reason', '')}")
            elif args.verbose:
                pdb_ok = "PDB" if result["has_protein_pdb"] else "---"
                sdf_ok = "SDF" if result["has_ligand_sdf"] else "---"
                logging.info(f"  {pdb_id}: {pdb_ok} {sdf_ok} [{result.get('ccd_code', '')}] "
                           f"{result.get('ligand_note', '')}")

        # Write evaluation CSV for this benchmark
        csv_path, n_rows = write_evaluation_csv(
            bench_results, pk_map, args.output_dir, benchmark
        )
        logging.info(f"  Eval CSV: {csv_path} ({n_rows} entries)")

        has_pdb = sum(1 for r in bench_results if r.get("has_protein_pdb"))
        has_sdf = sum(1 for r in bench_results if r.get("has_ligand_sdf"))
        no_sdf = sum(1 for r in bench_results
                     if r["pdb_id"].lower() in NO_SDF_ENTRIES and r["status"] == "ok")
        logging.info(f"  PDB: {has_pdb}/{len(entries)}, SDF: {has_sdf}/{len(entries)}, "
                    f"Peptide/PPI: {no_sdf}")

    # Write processing results CSV
    results_csv = os.path.join(args.output_dir, "processing_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames = [
        "pdb_id", "benchmark", "status", "ccd_code",
        "has_protein_pdb", "has_ligand_sdf", "ligand_note",
        "protein_pdb_path", "ligand_sdf_path",
    ]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(all_results)}")
    for status, count in sorted(totals.items()):
        print(f"  {status}: {count}")

    has_pdb = sum(1 for r in all_results if r.get("has_protein_pdb"))
    has_sdf = sum(1 for r in all_results if r.get("has_ligand_sdf"))
    print(f"\nProtein PDB: {has_pdb}")
    print(f"Ligand SDF:  {has_sdf}")
    print(f"\nResults: {results_csv}")
    print(f"Output:  {args.output_dir}/")

    for benchmark in benchmarks:
        csv_path = os.path.join(args.output_dir, f"{benchmark}_af3_test.csv")
        if os.path.exists(csv_path):
            print(f"\nAEV-PLIG eval CSV: {csv_path}")
            print(f"  Run: python process_and_predict.py \\")
            print(f"    --dataset_csv={csv_path} \\")
            print(f"    --data_name=af3_{benchmark} \\")
            print(f"    --trained_model_name=<MODEL> --device=0")


if __name__ == "__main__":
    main()
