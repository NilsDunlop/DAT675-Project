import os
import re
import csv
import json
import time
import logging
import argparse
from collections import OrderedDict
from difflib import SequenceMatcher

import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# ---------------------------------------------------------------------------
# Amino acid conversion table
# ---------------------------------------------------------------------------
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
    'MSE': 'M', 'HYP': 'P', 'SEP': 'S', 'TPO': 'T', 'CSO': 'C',
    'PTR': 'Y', 'MLY': 'K', 'KCX': 'K', 'CSS': 'C', 'CSD': 'C',
    # Non-standard residues found in CASF peptide ligands
    'OIC': 'P',   # (2S,3aS,7aS)-octahydroindole-2-carboxylic acid ~ Pro
    'AIB': 'A',   # alpha-aminoisobutyric acid ~ Ala
    'NME': 'G',   # N-methyl group (terminal)
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
MODEL_SEEDS = [42, 24, 19, 37, 51]
AF3_VERSION = 4
AF3_DIALECT = "alphafold3"
LOCAL_ALIGN_THRESHOLD = 0.80

# ---------------------------------------------------------------------------
# Solvent / cofactor / glycan / nucleotide exclusion list
#
# Items here are ALWAYS excluded -- they are never the benchmark ligand.
# For cofactors that might sometimes be the actual ligand (GDP, ADP, etc.),
# see CONDITIONAL_EXCLUDE_CCD below.
# ---------------------------------------------------------------------------
EXCLUDE_CCD = {
    "HOH", "WAT", "DOD",
    "GOL", "EDO", "PEG", "DMS", "SO4", "PO4", "CL", "NA", "MG", "ZN",
    "CA", "MN", "FE", "CO", "NI", "CU", "IOD", "BR", "FE2", "CU1",
    "K", "LI", "RB", "CS", "BA", "SR", "CD", "HG", "PT", "AU", "AG",
    "ACE", "ACT", "TRS", "EPE", "MES", "PIP", "IMD", "TAR",
    "MPD", "IPA", "EOH", "MOH", "EGL", "PGE", "PG4", "1PE",
    "DTT", "BME", "TCE", "DIO", "THF",
    "NAG", "MAN", "FUC", "GAL", "GLC", "SIA", "BGC", "BMA",
    "FCA", "NDG", "NGA",
    "HEM", "FAD", "FMN", "NAD", "NAP", "ATP", "AMP",
    "GTP", "COA", "SAM", "SAH",
}

# ---------------------------------------------------------------------------
# Conditional exclusion list
#
# These CCDs are common cofactors/nucleotides that are USUALLY crystallisation
# additives or co-substrates, but in some benchmark entries they ARE the
# measured ligand (e.g. GDP in GTPase binding assays).
#
# Resolution rule: exclude these ONLY if at least one other non-solvent,
# non-conditional CCD exists. If they are the sole non-trivial ligand,
# retain them.
# ---------------------------------------------------------------------------
CONDITIONAL_EXCLUDE_CCD = {
    "GDP", "ADP", "UDP", "CTP", "UTP", "CDP",
    "ACP",   # non-hydrolysable ATP analogue -- often additive
}

# ---------------------------------------------------------------------------
# Known additives / detergents / PEG variants
#
# Not excluded outright (they might be the only non-solvent CCD in rare cases),
# but strongly deprioritised during multi-candidate disambiguation.
# These are common crystallisation additives that appear as non-polymer
# entities in the PDB but are almost never the benchmark ligand.
# ---------------------------------------------------------------------------
KNOWN_ADDITIVE_CCD = {
    # PEG / polyethylene glycol variants
    "P6G", "PE4", "PE5", "PE8", "PG0", "2PE", "P33", "P4G",
    # Detergents
    "BOG", "LDA", "LMT", "OLC", "DMU", "DDQ", "SDS",
    # Buffer / small organic ions
    "FMT", "SCN", "NO3", "CO3", "CO2", "BCT", "MLI", "CAC", "CIT",
    "TLA", "FLC", "AKG", "PZE", "MRD",
    # Metal cofactors / exotic ions that slip through EXCLUDE_CCD
    "YB", "VN4", "MGF",
    # Amino acids as free ligands (usually crystallisation additives)
    "LEU", "ARG", "GLN",
    # Fatty acids
    "MYR", "PLM", "OLA", "STE",
    # Cholesterol (membrane protein additive)
    "CLR",
    # Monothioglycerol (reducing agent)
    "SGM",
}

# ---------------------------------------------------------------------------
# Manual CCD overrides
#
# Categories:
#   (a) Protonation state / tautomer cases (from CASF-2016, retained)
#   (b) Cofactor/nucleotide ligands retained per benchmark definition
#   (c) 0LigandBias entries identified during manual review
#
# Values: CCD code string, or None to exclude the entry entirely.
# ---------------------------------------------------------------------------
MANUAL_CCD_OVERRIDES = {
    # -- Protonation state / tautomer cases (CASF-2016) --------------------
    "1c5z": "BEN",
    "1q8u": "H52",
    "2xii": "TA9",
    "3d4z": "GIM",
    "3d6q": "U3S",
    "3dx1": "YHO",
    "3dx2": "MZB",
    "3ejr": "HN4",
    "3mss": "MS7",
    "3qqs": "17C",
    "3r88": "14F",
    "3twp": "SAL",
    "4djv": "0KM",
    "4ea2": "RWZ",
    "4gid": "0GH",
    "4gkm": "683",
    "4mme": "29Q",
    "4owm": "3F0",
    # -- Cofactor / nucleotide ligands retained per benchmark definition ---
    "1o0h": "ADP",   # CASF-2016: ADP is the measured ligand
    # -- 0LigandBias: GDP as actual measured ligand ------------------------
    # These are GTPase / GDP-binding protein entries where GDP is the ligand
    # with experimentally measured binding affinity. The conditional exclude
    # logic should handle most of these automatically, but we override
    # explicitly for entries that have other non-solvent CCDs that might
    # confuse the resolver.
    "1d2e": "GDP",   # Ras GTPase; MG is cofactor, GDP is measured ligand
    "1dar": "GDP",   # Ras-family GTPase; GDP is sole CCD after MG filtering
    "1r5n": "GDP",   # GTPase; GDP is sole ligand
    "1ryf": "GDP",   # GTPase; MG cofactor + GDP ligand
    "1tpz": "GDP",   # GTPase; MG + GDP + EDO + MPD (others are solvents)
    "1tq4": "GDP",   # GTPase; GDP + MG
    "3zy2": "GDP",   # GTPase; MN + GDP -- GDP is the ligand
    "6agp": "GDP",   # GTPase; GDP + MG
    # -- 0LigandBias: PRD / BIRD oligosaccharide entries -------------------
    # These entries have ligands classified as PRD (Biologically Interesting
    # Reference Dictionary) entries in RCSB -- typically oligosaccharides
    # (acarbose) that are not represented as single nonpolymer CCDs.
    # Since AF3 cannot take PRD codes directly, we exclude these entries
    # as they cannot be faithfully represented.
    "1gah": None,    # Ligand is acarbose (PRD_900007) -- no single CCD
    "1lf9": None,    # Ligand is acarbose (PRD_900007)
    "2zq0": None,    # Ligand is acarbose (PRD_900007)
    "3jyr": None,    # Ligand is acarbose (PRD_900007)
    "3jzj": None,    # Ligand is acarbose (PRD_900007)
    "1k1y": None,    # Ligand is acarbose (PRD_900007); TRS is buffer
    "2qmj": None,    # Ligand is acarbose (PRD_900007); NAG/GOL/SO4 additives
    # -- 0LigandBias: Protein-protein interaction entries ------------------
    # The "ligand" is another protein chain already present in the structure.
    # These are handled by PPI_OVERRIDES below — AF3 models both chains
    # natively as a multi-chain protein complex.
    # (1g1e and 1s5q: moved to PPI_OVERRIDES)
    # -- 0LigandBias: Modified residue in polymer chain --------------------
    # The "ligand" is a histone peptide with trimethylated lysine (M3L).
    # AF3 supports this via protein chain + ptmType modification.
    # (2l3r, 2l75, 4yhp: moved to PEPTIDE_LIGAND_OVERRIDES with modifications)
}

# ---------------------------------------------------------------------------
# Override notes -- free-text rationale for audit trail and methods section.
# ---------------------------------------------------------------------------
OVERRIDE_NOTES = {
    # Protonation state cases (CASF-2016) -- retained from v4
    "1c5z": "SDF formula C7H9N2 is benzamidinium (protonated); CCD BEN (C7H8N2) is the neutral form. Assigned BEN.",
    "1q8u": "SDF formula C16H22N3O2S is +1H relative to H52 (C16H21N3O2S). Assigned H52.",
    "2xii": "SDF formula C21H23N2O5 is +1H relative to TA9 (C21H22N2O5). Assigned TA9.",
    "3d4z": "SDF formula C8H12N2O4 is -1H relative to GIM (C8H13N2O4). Assigned GIM.",
    "3d6q": "SDF formula C14H22N3O5 is +1H relative to U3S (C14H21N3O5). Assigned U3S.",
    "3dx1": "SDF formula C5H12NO3 is +1H relative to YHO (C5H11NO3). Assigned YHO.",
    "3dx2": "SDF formula C6H15NO4S is +2H relative to MZB (C6H13NO4S). Assigned MZB.",
    "3ejr": "SDF formula C20H30NO4 is +1H relative to HN4 (C20H29NO4). Assigned HN4.",
    "3mss": "SDF formula C17H21N2O2 is +1H relative to MS7 (C17H20N2O2). Assigned MS7.",
    "3qqs": "SDF formula C14H9NO4 is -2H relative to 17C (C14H11NO4). Assigned 17C.",
    "3r88": "SDF formula C9H10NO4 is -1H relative to 14F (C9H11NO4). Assigned 14F.",
    "3twp": "SDF formula C7H5O3 is -1H relative to SAL (C7H6O3). Assigned SAL.",
    "4djv": "SDF formula C23H22N3O2 is +1H relative to 0KM (C23H21N3O2). Assigned 0KM.",
    "4ea2": "SDF formula C22H40N2 is +2H relative to RWZ (C22H38N2). Assigned RWZ.",
    "4gid": "SDF formula C35H48N5O6S is +1H relative to 0GH (C35H47N5O6S). Assigned 0GH.",
    "4gkm": "SDF formula C15H11NO4 is -2H relative to 683 (C15H13NO4). Assigned 683.",
    "4mme": "SDF formula C16H14ClN2O is +1H relative to 29Q (C16H13ClN2O). Assigned 29Q.",
    "4owm": "SDF formula C7H5FNO2 is -1H relative to 3F0 (C7H6FNO2). Assigned 3F0.",
    # Cofactor / nucleotide
    "1o0h": (
        "ADP (adenosine-5'-diphosphate) was filtered by the automated pipeline "
        "because ADP appears in EXCLUDE_CCD as a common cofactor. However, this "
        "entry is part of the official CASF-2016 benchmark and its binding "
        "affinity is experimentally measured. Re-included per benchmark definition."
    ),
    # Peptide ligands (CASF-2016)
    "1a30": (
        "CASF ligand is a tripeptide (GLU-ASP-LEU). RCSB classifies it as "
        "chain C (polymer entity). Provided as short protein chain with sequence EDL."
    ),
    "3bv9": (
        "CASF ligand is Thrombostatin FM (PRD_000276), a cyclic pentapeptide. "
        "Linearised sequence RPPAF provided as protein chain; non-natural "
        "residues approximated."
    ),
    "3uri": (
        "CASF ligand is DB5 peptide (PRD_000855), an octapeptide inhibitor. "
        "Sequence HPHLSAAH provided as protein chain."
    ),
    # 0LigandBias: GDP entries
    "1d2e": "GDP is the measured ligand in this Ras GTPase complex; MG is structural cofactor.",
    "1dar": "GDP is sole non-ion ligand; Ras-family GTPase.",
    "1r5n": "GDP is the sole ligand in this GTPase entry.",
    "1ryf": "GDP is the measured ligand; MG is structural cofactor.",
    "1tpz": "GDP is the measured ligand; MG, EDO, MPD are cofactor/additives.",
    "1tq4": "GDP is the measured ligand; MG is structural cofactor.",
    "3zy2": "GDP is the measured ligand; MN is structural cofactor.",
    "6agp": "GDP is the measured ligand; MG is structural cofactor.",
    # 0LigandBias: Excluded entries
    "1gah": "Ligand is acarbose (PRD_900007), an oligosaccharide with no single CCD representation suitable for AF3.",
    "1lf9": "Ligand is acarbose (PRD_900007); excluded.",
    "2zq0": "Ligand is acarbose (PRD_900007); excluded.",
    "3jyr": "Ligand is acarbose (PRD_900007); excluded.",
    "3jzj": "Ligand is acarbose (PRD_900007); excluded.",
    "1k1y": "Ligand is acarbose (PRD_900007); TRS is buffer. Excluded.",
    "2qmj": "Ligand is acarbose (PRD_900007); NAG/GOL/SO4 are additives. Excluded.",
    "1g1e": (
        "PPI entry: MAD1-SIN3A interaction. Both chains are protein entities "
        "already present in the PDB. AF3 JSON includes both as protein chains "
        "with no additional ligand entity."
    ),
    "1s5q": (
        "PPI entry: MAD-SIN3A interaction. Both chains are protein entities "
        "already present in the PDB. AF3 JSON includes both as protein chains "
        "with no additional ligand entity."
    ),
    "4yhp": (
        "Ligand is H3K9me3 peptide: histone H3 tail with trimethylated K9. "
        "Provided as short protein chain with ptmType M3L at the lysine position. "
        "Sequence ARTKQTARKSTGGKAP from RCSB chain P."
    ),
    "2l3r": (
        "Ligand is histone H3 K9me3 peptide bound to UHRF1 Tudor domain. "
        "Provided as short protein chain ARTKQTARKST with ptmType M3L at position 9."
    ),
    "2l75": (
        "Ligand is histone H3 K9me3 peptide bound to CHD4 chromodomain. "
        "Provided as short protein chain ARTKQTARKSTGGY with ptmType M3L at position 9."
    ),
    # 0LigandBias: Leupeptin entries
    "1jrs": "Ligand is leupeptin (PRD_000216). RCSB FASTA: XLLR. Provided as ALLR (X=N-acetyl->Ala, Arg-al->Arg).",
    "1tl9": "Ligand is leupeptin (PRD_000216). RCSB FASTA: XLLR. Provided as ALLR.",
    "1pop": "Ligand is leupeptin (PRD_000216). RCSB FASTA: XLLR. Provided as ALLR.",
}

# ---------------------------------------------------------------------------
# Peptide ligand overrides
# ---------------------------------------------------------------------------
PEPTIDE_LIGAND_OVERRIDES = {
    # CASF-2016
    "1a30": {
        "sequence": "EDL",
        "description": "Tripeptide ligand GLU-ASP-LEU from CASF-2016 (PDB: 1A30)",
        "note": "CASF ligand is C-terminal tripeptide of the inhibitor",
    },
    "3bv9": {
        "sequence": "RPPAF",
        "description": (
            "Cyclic peptide ligand PRD_000276 (Thrombostatin FM) from PDB 3BV9. "
            "Non-natural residues approximated: Oic->P, D-Arg->R, D-Ala->A, p-Me-Phe->F."
        ),
        "note": "Cyclic peptide; linearised with non-natural residues approximated",
    },
    "3uri": {
        "sequence": "HPHLSAAH",
        "description": "Peptide ligand PRD_000855 (DB5 peptide) from PDB 3URI.",
        "note": "Non-natural residue at position 6 approximated as Ala",
    },
    # 0LigandBias: Leupeptin entries
    # Leupeptin = Ac-Leu-Leu-Arg-al (N-acetyl-L-leucyl-L-leucyl-L-arginal)
    # RCSB FASTA shows sequence as XLLR where X = N-acetyl cap.
    # The C-terminal is arginal (Arg with aldehyde); AF3 receives standard Arg.
    # The N-terminal X (N-acetyl) is approximated as Ala.
    # Final sequence: ALLR (matching RCSB residue order: X-Leu-Leu-Arg).
    "1jrs": {
        "sequence": "ALLR",
        "description": (
            "Leupeptin (PRD_000216) from PDB 1JRS. RCSB FASTA: XLLR. "
            "N-acetyl cap (X) approximated as Ala; C-terminal arginal as Arg."
        ),
        "note": "Leupeptin: XLLR -> ALLR (X=N-acetyl->A, Arg-al->R)",
    },
    "1tl9": {
        "sequence": "ALLR",
        "description": (
            "Leupeptin (PRD_000216) from PDB 1TL9. RCSB FASTA: XLLR. "
            "N-acetyl cap (X) approximated as Ala; C-terminal arginal as Arg."
        ),
        "note": "Leupeptin: XLLR -> ALLR (X=N-acetyl->A, Arg-al->R)",
    },
    "1pop": {
        "sequence": "ALLR",
        "description": (
            "Leupeptin (PRD_000216) from PDB 1POP. RCSB FASTA: XLLR. "
            "N-acetyl cap (X) approximated as Ala; C-terminal arginal as Arg."
        ),
        "note": "Leupeptin: XLLR -> ALLR (X=N-acetyl->A, Arg-al->R)",
    },
    # 0LigandBias: Histone H3 K9me3 peptide entries
    # The "ligand" is a histone H3 tail peptide with trimethylated lysine 9.
    # AF3 supports protein modifications via ptmType in the modifications field.
    # M3L (N-TRIMETHYLLYSINE) is a valid CCD code for the PTM.
    # The histone peptide sequence is from the RCSB FASTA for each entry.
    "2l3r": {
        "sequence": "ARTKQTARKST",
        "description": (
            "Histone H3 K9me3 peptide from PDB 2L3R. Chain B (11-mer). "
            "Trimethylated lysine at position 9 specified via ptmType M3L."
        ),
        "note": "H3K9me3 peptide; K9 -> M3L modification",
        "modifications": [{"ptmType": "M3L", "ptmPosition": 9}],
    },
    "2l75": {
        "sequence": "ARTKQTARKSTGGY",
        "description": (
            "Histone H3 K9me3 peptide from PDB 2L75. Chain B (14-mer). "
            "Trimethylated lysine at position 9 specified via ptmType M3L."
        ),
        "note": "H3K9me3 peptide; K9 -> M3L modification",
        "modifications": [{"ptmType": "M3L", "ptmPosition": 9}],
    },
    "4yhp": {
        "sequence": "ARTKQTARKSTGGKAP",
        "description": (
            "Histone H3 K9me3 peptide from PDB 4YHP. Chain P (16-mer). "
            "Trimethylated lysine at position 9 specified via ptmType M3L."
        ),
        "note": "H3K9me3 peptide; K9 -> M3L modification",
        "modifications": [{"ptmType": "M3L", "ptmPosition": 9}],
    },
}

# ---------------------------------------------------------------------------
# PPI (protein-protein interaction) overrides
#
# Entries where the benchmark "ligand" is actually another protein chain.
# The benchmark PDB file typically only contains the receptor; the "ligand"
# peptide chain must be provided explicitly from the RCSB FASTA.
# These are handled identically to PEPTIDE_LIGAND_OVERRIDES in build_af3_json
# (added as an additional protein chain).
#
# Format: {pdb_id: {"sequence": "...", "description": "...", "note": "..."}}
# ---------------------------------------------------------------------------
PPI_OVERRIDES = {
    "1g1e": {
        # RCSB FASTA: >1G1E_1|Chain A|MAD1 PROTEIN|null
        # RMNIQMLLEAADYLER
        "sequence": "RMNIQMLLEAADYLER",
        "description": (
            "MAD1 protein (chain A) from PDB 1G1E. 16-residue peptide that "
            "binds SIN3A. Provided as additional protein chain for PPI modelling."
        ),
        "note": "PPI: MAD1 peptide binds SIN3A; sequence from RCSB FASTA chain A",
    },
    "1s5q": {
        # RCSB FASTA: >1S5Q_1|Chain A|MAD protein|null
        # RMNIQMLLEAADYLER
        "sequence": "RMNIQMLLEAADYLER",
        "description": (
            "MAD protein (chain A) from PDB 1S5Q. 16-residue peptide that "
            "binds SIN3A. Provided as additional protein chain for PPI modelling."
        ),
        "note": "PPI: MAD peptide binds SIN3A; sequence from RCSB FASTA chain A",
    },
}

# ---------------------------------------------------------------------------
# GraphQL query
# ---------------------------------------------------------------------------
GRAPHQL_QUERY = """
query GetEntryData($pdb_id: String!) {
  entry(entry_id: $pdb_id) {
    rcsb_id
    polymer_entities {
      rcsb_id
      entity_poly {
        pdbx_seq_one_letter_code_can
        type
      }
      polymer_entity_instances {
        rcsb_id
        rcsb_polymer_entity_instance_container_identifiers {
          auth_asym_id
          asym_id
        }
      }
    }
    nonpolymer_entities {
      rcsb_id
      nonpolymer_comp {
        chem_comp {
          id
          name
          formula
          formula_weight
        }
      }
      nonpolymer_entity_instances {
        rcsb_id
        rcsb_nonpolymer_entity_instance_container_identifiers {
          auth_asym_id
          asym_id
        }
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# PDB file parsing
# ---------------------------------------------------------------------------

def parse_seqres(pdb_path):
    chains = OrderedDict()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("SEQRES"):
                continue
            chain_id = line[11].strip()
            num_res = int(line[13:17].strip())
            residues = line[19:].split()
            if chain_id not in chains:
                chains[chain_id] = {"declared_length": num_res, "residues": []}
            chains[chain_id]["residues"].extend(residues)
    return chains


def parse_atom_sequences(pdb_path):
    chains = OrderedDict()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip()
            res_name = line[17:20].strip()
            res_num = int(line[22:26].strip())
            if chain_id not in chains:
                chains[chain_id] = OrderedDict()
            chains[chain_id][res_num] = res_name
    result = OrderedDict()
    for cid, residues in chains.items():
        res_list = [residues[k] for k in sorted(residues.keys())]
        result[cid] = {"declared_length": len(res_list), "residues": res_list}
    return result


def residues_to_sequence(residues_3letter):
    seq, unknowns = [], set()
    for r in residues_3letter:
        one = THREE_TO_ONE.get(r)
        if one is None:
            unknowns.add(r)
            one = 'X'
        seq.append(one)
    return ''.join(seq), unknowns


# ---------------------------------------------------------------------------
# SDF parsing -- RDKit-based formula and MW computation
# ---------------------------------------------------------------------------

def _compute_formula_rdkit(sdf_path):
    """
    Use RDKit to parse the SDF and compute molecular formula and MW.

    Returns (formula_hill_str, mw_float) or (None, None) on failure.
    The formula is normalised to Hill notation via our own normaliser
    to ensure consistency with the API-side formulas.
    """
    try:
        mol = Chem.MolFromMolFile(sdf_path, sanitize=True, removeHs=False)
        if mol is None:
            # Retry without sanitization for problematic files
            mol = Chem.MolFromMolFile(sdf_path, sanitize=False, removeHs=False)
        if mol is None:
            return None, None
        raw_formula = rdMolDescriptors.CalcMolFormula(mol)
        mw = rdMolDescriptors.CalcExactMolWt(mol)
        # Normalise through our Hill formatter for consistent comparison
        hill = normalise_formula(raw_formula)
        return hill, mw
    except Exception as e:
        logging.debug(f"RDKit formula computation failed for {sdf_path}: {e}")
        return None, None


def parse_sdf_ligand_info(sdf_path):
    """
    Parse SDF file for mol name, molecular formula, and molecular weight.

    Resolution order for formula:
      1. > <MOLECULAR_FORMULA> SDF tag (if present)
      2. RDKit: Chem.MolFromMolFile + CalcMolFormula (handles implicit H)

    Returns (mol_name, molecular_formula, formula_source, sdf_mw)
    where:
      formula_source: "sdf_tag", "rdkit", or None
      sdf_mw: float molecular weight from RDKit, or None
    """
    mol_name = None
    molecular_formula = None
    formula_source = None
    sdf_mw = None
    try:
        with open(sdf_path) as f:
            lines = f.readlines()
        if not lines:
            return None, None, None, None

        # Mol name from line 1
        candidate = lines[0].strip()
        if candidate and len(candidate) <= 6 and candidate.isalnum():
            mol_name = candidate.upper()

        # Look for MOLECULAR_FORMULA tag
        for i, line in enumerate(lines):
            tag = line.strip().upper().replace(" ", "")
            if tag in ("><MOLECULAR_FORMULA>", "> <MOLECULAR_FORMULA>",
                       "><FORMULA>", "> <FORMULA>"):
                if i + 1 < len(lines):
                    formula = lines[i + 1].strip()
                    if formula:
                        molecular_formula = formula.replace(" ", "")
                        formula_source = "sdf_tag"
                        break

        # Always try RDKit for MW (and formula if tag was missing)
        rdkit_formula, rdkit_mw = _compute_formula_rdkit(sdf_path)
        if rdkit_mw is not None:
            sdf_mw = rdkit_mw

        if not molecular_formula and rdkit_formula:
            molecular_formula = rdkit_formula
            formula_source = "rdkit"

    except Exception as e:
        logging.debug(f"SDF parse error for {sdf_path}: {e}")

    return mol_name, molecular_formula, formula_source, sdf_mw


# ---------------------------------------------------------------------------
# Formula normalisation -- Hill notation
# ---------------------------------------------------------------------------

def parse_formula_to_dict(formula):
    if not formula:
        return {}
    f = formula.replace(" ", "")
    f = re.sub(r'[\d]*[+-]$', '', f)
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', f)
    counts = {}
    for element, count in tokens:
        if not element:
            continue
        n = int(count) if count else 1
        counts[element] = counts.get(element, 0) + n
    return counts


def dict_to_hill(counts):
    if not counts:
        return ""
    result = ""
    remaining = dict(counts)
    if 'C' in remaining:
        n = remaining.pop('C')
        result += f"C{n}" if n > 1 else "C"
    if 'H' in remaining:
        n = remaining.pop('H')
        result += f"H{n}" if n > 1 else "H"
    for element in sorted(remaining.keys()):
        n = remaining[element]
        result += f"{element}{n}" if n > 1 else element
    return result


def normalise_formula(formula):
    return dict_to_hill(parse_formula_to_dict(formula))


def formula_atom_diff(formula_a, formula_b):
    a = parse_formula_to_dict(formula_a)
    b = parse_formula_to_dict(formula_b)
    all_elements = set(a.keys()) | set(b.keys())
    diffs = {}
    for el in all_elements:
        diff = a.get(el, 0) - b.get(el, 0)
        if diff != 0:
            diffs[el] = diff
    total_abs = sum(abs(v) for v in diffs.values())
    return diffs, total_abs


def _format_atom_diff(diffs):
    parts = []
    for el, d in sorted(diffs.items()):
        sign = "+" if d > 0 else ""
        parts.append(f"{sign}{d}{el}")
    return ", ".join(parts) if parts else "none"


def formula_to_weight_approx(formula):
    """Approximate molecular weight from formula string (for disambiguation)."""
    ATOMIC_WEIGHTS = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81,
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
        'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
        'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546,
        'Zn': 65.38, 'As': 74.922, 'Se': 78.96, 'Br': 79.904, 'Rb': 85.468,
        'Sr': 87.62, 'Mo': 95.96, 'Ag': 107.868, 'Cd': 112.411, 'Sn': 118.710,
        'I': 126.904, 'Cs': 132.905, 'Ba': 137.327, 'Au': 196.967,
        'Hg': 200.59, 'Pt': 195.084, 'Yb': 173.045, 'V': 50.942,
    }
    counts = parse_formula_to_dict(formula)
    mw = 0.0
    for el, n in counts.items():
        mw += ATOMIC_WEIGHTS.get(el, 100.0) * n
    return mw


# ---------------------------------------------------------------------------
# RCSB PDB API
# ---------------------------------------------------------------------------

def query_pdb_api(pdb_id, delay=0.25, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": GRAPHQL_QUERY, "variables": {"pdb_id": pdb_id.upper()}},
                timeout=30,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                logging.warning(f"{pdb_id}: GraphQL errors: {data['errors']}")
                return None
            time.sleep(delay)
            return data.get("data", {}).get("entry")
        except requests.RequestException as e:
            logging.warning(f"{pdb_id}: API request failed (attempt {attempt+1}): {e}")
            time.sleep(delay * (attempt + 1) * 2)
    return None


def parse_api_response(entry_data):
    chain_sequences = {}
    ligand_info = []
    if not entry_data:
        return chain_sequences, ligand_info
    for entity in entry_data.get("polymer_entities") or []:
        entity_poly = entity.get("entity_poly") or {}
        seq = entity_poly.get("pdbx_seq_one_letter_code_can", "")
        seq = seq.replace("\n", "").replace(" ", "") if seq else ""
        entity_type = entity_poly.get("type", "")
        if "polypeptide" not in entity_type.lower():
            continue
        for instance in entity.get("polymer_entity_instances") or []:
            container = instance.get(
                "rcsb_polymer_entity_instance_container_identifiers", {}
            ) or {}
            chain_id = container.get("auth_asym_id", "").strip()
            if chain_id and seq:
                chain_sequences[chain_id] = seq
    for entity in entry_data.get("nonpolymer_entities") or []:
        comp = (entity.get("nonpolymer_comp") or {}).get("chem_comp") or {}
        ccd_id = comp.get("id", "").strip()
        if ccd_id:
            raw_formula = comp.get("formula", "")
            ligand_info.append({
                "ccd_id": ccd_id,
                "name": comp.get("name", ""),
                "formula": raw_formula,
                "formula_hill": normalise_formula(raw_formula),
                "formula_weight": comp.get("formula_weight"),
            })
    return chain_sequences, ligand_info


# ---------------------------------------------------------------------------
# Sequence comparison
# ---------------------------------------------------------------------------

def local_alignment_coverage(casf_seq, pdb_seq):
    if not casf_seq:
        return 0.0
    matcher = SequenceMatcher(None, casf_seq, pdb_seq, autojunk=False)
    matched = sum(t.size for t in matcher.get_matching_blocks() if t.size > 0)
    return matched / len(casf_seq)


def compare_sequences(casf_seq, pdb_seq):
    if casf_seq == pdb_seq:
        return 1.0, True, "identical"
    if casf_seq in pdb_seq:
        coverage = len(casf_seq) / len(pdb_seq)
        return coverage, True, (
            f"CASF is contiguous substring of PDB canonical "
            f"(CASF={len(casf_seq)}, PDB={len(pdb_seq)}, coverage={coverage:.2f})"
        )
    coverage = local_alignment_coverage(casf_seq, pdb_seq)
    if coverage >= LOCAL_ALIGN_THRESHOLD:
        return coverage, True, (
            f"CASF locally aligns to PDB canonical at {coverage:.2f} coverage "
            f"(CASF={len(casf_seq)}, PDB={len(pdb_seq)}) -- "
            f"accepted as crystallographic truncation [threshold={LOCAL_ALIGN_THRESHOLD}]"
        )
    return coverage, False, (
        f"sequence mismatch: local alignment coverage={coverage:.2f} "
        f"(below threshold {LOCAL_ALIGN_THRESHOLD}), "
        f"CASF={len(casf_seq)}, PDB={len(pdb_seq)} -- flagged for review"
    )


# ---------------------------------------------------------------------------
# Missing chain resolution
# ---------------------------------------------------------------------------

def resolve_missing_chains(casf_sequences, api_chain_seqs, pdb_id):
    resolved_chains = OrderedDict()
    chain_notes = []
    replicated_chains = set()
    has_unresolvable = False

    for chain_id, casf_seq in casf_sequences.items():
        if chain_id in api_chain_seqs:
            resolved_chains[chain_id] = api_chain_seqs[chain_id]
            continue
        best_cov = 0.0
        best_api_seq = None
        best_api_chain = None
        for api_cid, api_seq in api_chain_seqs.items():
            cov = local_alignment_coverage(casf_seq, api_seq)
            if cov > best_cov:
                best_cov = cov
                best_api_seq = api_seq
                best_api_chain = api_cid
        if best_api_seq is not None and best_cov >= LOCAL_ALIGN_THRESHOLD:
            resolved_chains[chain_id] = best_api_seq
            replicated_chains.add(chain_id)
            chain_notes.append(
                f"chain {chain_id}: not in PDB API, resolved as homo-oligomer "
                f"copy of chain {best_api_chain} "
                f"(local alignment coverage={best_cov:.2f})"
            )
        else:
            has_unresolvable = True
            chain_notes.append(
                f"chain {chain_id}: not in PDB API and sequence does not match "
                f"any verified chain "
                f"(best local alignment coverage={best_cov:.2f}, "
                f"threshold={LOCAL_ALIGN_THRESHOLD}) -- flagged"
            )

    return resolved_chains, chain_notes, replicated_chains, has_unresolvable


# ---------------------------------------------------------------------------
# CCD code resolution -- enhanced with conditional exclusion and MW heuristic
# ---------------------------------------------------------------------------

def _find_close_formula_candidates(sdf_formula_hill, candidates, max_diff=2):
    close = []
    for li in candidates:
        diffs, total = formula_atom_diff(sdf_formula_hill, li["formula_hill"])
        if 0 < total <= max_diff:
            close.append({
                "ccd_id": li["ccd_id"],
                "name": li["name"],
                "formula_hill": li["formula_hill"],
                "formula_weight": li.get("formula_weight"),
                "diff_str": _format_atom_diff(diffs),
                "total_diff": total,
            })
    close.sort(key=lambda x: x["total_diff"])
    return close


def _build_review_data(pdb_id, sdf_mol_name, sdf_formula_hill,
                       candidates, reason, close_candidates=None,
                       all_ccd_codes=None):
    return {
        "pdb_id": pdb_id,
        "sdf_mol_name": sdf_mol_name or "",
        "sdf_formula_hill": sdf_formula_hill or "",
        "candidates": candidates,
        "reason": reason,
        "close_candidates": close_candidates or [],
        "all_ccd_codes": all_ccd_codes or [],
        "rcsb_entry_url": f"https://www.rcsb.org/structure/{pdb_id.upper()}",
    }


def _filter_ligands(ligand_info):
    """
    Apply tiered filtering to ligand info list.

    Returns:
        non_solvent:     list after removing EXCLUDE_CCD items
        primary:         list after also removing CONDITIONAL_EXCLUDE_CCD
                         (only if non-conditional candidates exist)
        additive_free:   list after also removing KNOWN_ADDITIVE_CCD
                         (only if non-additive candidates exist)
    """
    # Tier 1: Remove always-excluded
    non_solvent = [li for li in ligand_info if li["ccd_id"] not in EXCLUDE_CCD]

    # Tier 2: Conditionally remove cofactor/nucleotide CCDs
    non_conditional = [li for li in non_solvent
                       if li["ccd_id"] not in CONDITIONAL_EXCLUDE_CCD]
    if non_conditional:
        primary = non_conditional
    else:
        # All non-solvent CCDs are conditional -- keep them
        primary = non_solvent

    # Tier 3: Remove known additives if better candidates exist
    non_additive = [li for li in primary
                    if li["ccd_id"] not in KNOWN_ADDITIVE_CCD]
    if non_additive:
        additive_free = non_additive
    else:
        additive_free = primary

    return non_solvent, primary, additive_free


def _select_by_mw_heuristic(candidates, sdf_formula_hill, sdf_mw=None):
    """
    Among multiple candidates, select the one most likely to be the
    drug-like / bioactive ligand using a molecular weight heuristic.

    Preference order:
      1. If sdf_mw is provided (from RDKit), pick the candidate closest in MW
      2. If SDF formula MW can be approximated, pick closest
      3. Otherwise, pick the highest-MW candidate

    Returns (selected_ligand_info_dict, resolution_note_str)
    """
    if not candidates:
        return None, ""

    if len(candidates) == 1:
        return candidates[0], ""

    # Use RDKit MW if available, else approximate from formula
    if sdf_mw is None or sdf_mw <= 0:
        sdf_mw = formula_to_weight_approx(sdf_formula_hill) if sdf_formula_hill else 0.0

    if sdf_mw > 50:
        # Match by closest MW to SDF
        best = min(candidates,
                   key=lambda li: abs((li.get("formula_weight") or
                                       formula_to_weight_approx(li["formula_hill"])) - sdf_mw))
        best_mw = best.get('formula_weight') or formula_to_weight_approx(best['formula_hill'])
        return best, (
            f"selected {best['ccd_id']} by closest MW to SDF "
            f"(SDF~{sdf_mw:.0f}, {best['ccd_id']}~{best_mw:.0f}); "
        )
    else:
        # No SDF MW -- pick largest
        best = max(candidates,
                   key=lambda li: li.get("formula_weight") or
                                  formula_to_weight_approx(li["formula_hill"]))
        best_mw = best.get('formula_weight') or formula_to_weight_approx(best['formula_hill'])
        return best, (
            f"selected {best['ccd_id']} by highest MW heuristic "
            f"(MW~{best_mw:.0f}); "
        )


def resolve_ccd_code(pdb_id, ligand_info, sdf_mol_name, sdf_formula, sdf_mw=None):
    """
    Determine the CCD code for the AF3 JSON.
    All formula comparisons use Hill-normalised strings.
    sdf_mw: molecular weight from RDKit (if available) for better disambiguation.
    Returns (ccd_code or None, flag_reason str, resolution_method str, review_data dict).
    """
    sdf_formula_hill = normalise_formula(sdf_formula) if sdf_formula else ""
    all_ccd_codes = [li["ccd_id"] for li in ligand_info]

    # Apply tiered filtering
    non_solvent, primary, additive_free = _filter_ligands(ligand_info)
    non_solvent_ids = [li["ccd_id"] for li in non_solvent]
    primary_ids = [li["ccd_id"] for li in primary]
    resolution_method = ""

    # Priority 1: SDF mol name exact match (in any API CCD)
    if sdf_mol_name and sdf_mol_name in all_ccd_codes:
        resolution_method = "SDF_mol_name_exact_match"
        if sdf_formula_hill:
            matched_li = next(
                (li for li in ligand_info if li["ccd_id"] == sdf_mol_name), None
            )
            if matched_li:
                api_f = matched_li["formula_hill"]
                if api_f and api_f == sdf_formula_hill:
                    resolution_method = "SDF_mol_name_exact_match+formula_confirmed"
                elif api_f:
                    diffs, total = formula_atom_diff(sdf_formula_hill, api_f)
                    diff_str = _format_atom_diff(diffs)
                    if total <= 2:
                        resolution_method = "SDF_mol_name_exact_match+formula_warning"
                        return sdf_mol_name, (
                            f"SDF mol name '{sdf_mol_name}' matched; Hill formula "
                            f"differs by {diff_str} (protonation/tautomer); "
                        ), resolution_method, {}
                    else:
                        return None, (
                            f"SDF mol name '{sdf_mol_name}' matched API CCD but "
                            f"Hill formula differs by {diff_str} -- manual review needed"
                        ), resolution_method, _build_review_data(
                            pdb_id, sdf_mol_name, sdf_formula_hill,
                            non_solvent, "name_matched_formula_large_diff"
                        )
        return sdf_mol_name, "", resolution_method, {}

    # Priority 2: SDF mol name not in API -- formula rescue
    if sdf_mol_name and sdf_mol_name not in all_ccd_codes:
        if sdf_formula_hill:
            matches = [li for li in non_solvent if li["formula_hill"] == sdf_formula_hill]
            if len(matches) == 1:
                resolution_method = "SDF_formula_rescue_after_name_miss"
                return matches[0]["ccd_id"], (
                    f"SDF mol name '{sdf_mol_name}' not in API; rescued via "
                    f"Hill formula '{sdf_formula_hill}' -> {matches[0]['ccd_id']}; "
                ), resolution_method, {}
            elif len(matches) > 1:
                return None, (
                    f"SDF mol name '{sdf_mol_name}' not in API; formula ambiguous "
                    f"across: {','.join(li['ccd_id'] for li in matches)}"
                ), "", _build_review_data(
                    pdb_id, sdf_mol_name, sdf_formula_hill,
                    non_solvent, "name_miss_formula_ambiguous"
                )
        # No formula rescue -- fall through to general resolution

    # Priority 3: Formula-based disambiguation among additive-free candidates
    if sdf_formula_hill and len(additive_free) > 1:
        matches = [li for li in additive_free if li["formula_hill"] == sdf_formula_hill]
        if len(matches) == 1:
            resolution_method = "SDF_formula_disambiguation"
            return matches[0]["ccd_id"], (
                f"multiple non-solvent ligands resolved via Hill formula "
                f"'{sdf_formula_hill}' -> {matches[0]['ccd_id']}; "
            ), resolution_method, {}
        elif len(matches) > 1:
            best, note = _select_by_mw_heuristic(matches, sdf_formula_hill, sdf_mw)
            if best:
                resolution_method = "SDF_formula_disambiguation+MW_heuristic"
                return best["ccd_id"], note, resolution_method, {}

        # No exact match -- try close matches (+/-2 atoms, protonation)
        close = _find_close_formula_candidates(sdf_formula_hill, additive_free, max_diff=2)
        if len(close) == 1:
            resolution_method = "SDF_formula_close_match"
            return close[0]["ccd_id"], (
                f"Hill formula '{sdf_formula_hill}' close match to "
                f"{close[0]['ccd_id']} (diff: {close[0]['diff_str']}); "
            ), resolution_method, {}

    # Priority 3b: Formula disambiguation among all primary candidates
    if sdf_formula_hill and len(primary) > 1:
        matches = [li for li in primary if li["formula_hill"] == sdf_formula_hill]
        if len(matches) == 1:
            resolution_method = "SDF_formula_disambiguation_primary"
            return matches[0]["ccd_id"], (
                f"resolved via Hill formula among primary candidates "
                f"'{sdf_formula_hill}' -> {matches[0]['ccd_id']}; "
            ), resolution_method, {}
        close = _find_close_formula_candidates(sdf_formula_hill, primary, max_diff=2)
        if len(close) == 1:
            resolution_method = "SDF_formula_close_match_primary"
            return close[0]["ccd_id"], (
                f"Hill formula '{sdf_formula_hill}' close match to "
                f"{close[0]['ccd_id']} (diff: {close[0]['diff_str']}); "
            ), resolution_method, {}

    # Priority 4: Exactly one additive-free candidate
    if len(additive_free) == 1:
        li = additive_free[0]
        resolution_method = "sole_non_additive_ligand"
        note = ""
        if sdf_formula_hill:
            api_f = li["formula_hill"]
            if api_f and api_f == sdf_formula_hill:
                resolution_method = "sole_non_additive_ligand+formula_confirmed"
            elif api_f:
                diffs, total = formula_atom_diff(sdf_formula_hill, api_f)
                diff_str = _format_atom_diff(diffs)
                note = (
                    f"Hill formula differs by {diff_str} "
                    f"(SDF: {sdf_formula_hill}, API: {api_f}) -- "
                    f"{'likely protonation/tautomer; ' if total <= 2 else 'proceeding with warning; '}"
                )
        else:
            note = f"no SDF formula, used sole non-additive ligand '{li['ccd_id']}'; "
        return li["ccd_id"], note, resolution_method, {}

    # Priority 5: Exactly one primary candidate (after conditional exclusion)
    if len(primary) == 1:
        li = primary[0]
        resolution_method = "sole_primary_ligand"
        note = ""
        if sdf_formula_hill:
            api_f = li["formula_hill"]
            if api_f and api_f == sdf_formula_hill:
                resolution_method = "sole_primary_ligand+formula_confirmed"
            elif api_f:
                diffs, total = formula_atom_diff(sdf_formula_hill, api_f)
                diff_str = _format_atom_diff(diffs)
                note = (
                    f"Hill formula differs by {diff_str} -- "
                    f"{'likely protonation/tautomer; ' if total <= 2 else 'proceeding with warning; '}"
                )
        else:
            note = f"no SDF formula, used sole primary ligand '{li['ccd_id']}'; "
        return li["ccd_id"], note, resolution_method, {}

    # Priority 6: Exactly one non-solvent candidate
    if len(non_solvent) == 1:
        li = non_solvent[0]
        resolution_method = "sole_non_solvent_ligand"
        note = ""
        if sdf_formula_hill:
            api_f = li["formula_hill"]
            if api_f and api_f == sdf_formula_hill:
                resolution_method = "sole_non_solvent_ligand+formula_confirmed"
            elif api_f:
                diffs, total = formula_atom_diff(sdf_formula_hill, api_f)
                diff_str = _format_atom_diff(diffs)
                note = (
                    f"Hill formula differs by {diff_str} "
                    f"(SDF: {sdf_formula_hill}, API: {api_f}) -- "
                    f"{'likely protonation/tautomer; ' if total <= 2 else 'proceeding with warning; '}"
                )
        else:
            note = f"no SDF formula, used sole API ligand '{li['ccd_id']}'; "
        return li["ccd_id"], note, resolution_method, {}

    # Priority 7: MW-based heuristic for multiple additive-free candidates
    if len(additive_free) > 1:
        best, note = _select_by_mw_heuristic(additive_free, sdf_formula_hill, sdf_mw)
        if best:
            resolution_method = "MW_heuristic_additive_free"
            return best["ccd_id"], note, resolution_method, {}

    # Priority 8: MW-based heuristic for multiple primary candidates
    if len(primary) > 1:
        best, note = _select_by_mw_heuristic(primary, sdf_formula_hill, sdf_mw)
        if best:
            resolution_method = "MW_heuristic_primary"
            return best["ccd_id"], note, resolution_method, {}

    # Priority 9: Zero non-solvent ligands
    if len(non_solvent) == 0:
        return None, (
            f"no non-solvent ligand CCD found after filtering "
            f"(all API CCDs: {','.join(all_ccd_codes) or 'none'})"
        ), "", _build_review_data(
            pdb_id, sdf_mol_name, sdf_formula_hill, [],
            "no_non_solvent_ligand", all_ccd_codes=all_ccd_codes
        )

    # Priority 10: Multiple candidates, no resolution path
    close = _find_close_formula_candidates(sdf_formula_hill, non_solvent, max_diff=2)
    return None, (
        f"multiple non-solvent ligands, no hint: "
        f"{','.join(non_solvent_ids)} -- manual review needed"
    ), "", _build_review_data(
        pdb_id, sdf_mol_name, sdf_formula_hill, non_solvent, "multiple_no_hint",
        close_candidates=close
    )


# ---------------------------------------------------------------------------
# AF3 JSON builder
# ---------------------------------------------------------------------------

def build_af3_json(pdb_id, chain_sequences, ccd_code,
                   replicated_chains=None, peptide_override=None,
                   benchmark="casf2016"):
    """
    Build AF3 JSON.

    Identical protein chains (same sequence) are deduplicated into a single
    entry with "id": ["A", "B", ...] per the AF3 input spec.

    If peptide_override is provided (dict with 'sequence', 'description',
    and optional 'modifications'), the ligand is added as an additional
    protein chain. This is used for peptide ligands, H3K9me3 PTM entries,
    and PPI entries where the "ligand" is a protein chain.
    Otherwise, ccd_code is used as a standard CCD ligand.
    """
    if replicated_chains is None:
        replicated_chains = set()

    # -- Build protein chain entries, deduplicating identical sequences -----
    # Group chain IDs by sequence to detect homo-oligomers.
    seq_to_ids = OrderedDict()   # sequence -> list of assigned chain letters
    chain_letters = [chr(ord('A') + i) for i in range(len(chain_sequences))]

    for new_id, (orig_id, seq) in zip(chain_letters, chain_sequences.items()):
        seq_to_ids.setdefault(seq, []).append(new_id)

    sequences = []
    for seq, ids in seq_to_ids.items():
        if len(ids) == 1:
            chain_id_field = ids[0]
            desc = f"Chain {ids[0]} from PDB {pdb_id.upper()}"
        else:
            chain_id_field = ids
            desc = f"Chains {','.join(ids)} from PDB {pdb_id.upper()}"
        sequences.append({
            "protein": {"id": chain_id_field, "sequence": seq,
                        "description": desc}
        })

    # -- Add ligand / peptide -----------------------------------------------
    # Next available chain letter after all protein chains
    next_id = chr(ord('A') + len(chain_sequences))

    if peptide_override:
        # Peptide / PPI ligand: add as short protein chain, with optional PTM
        protein_entry = {
            "id": next_id,
            "sequence": peptide_override["sequence"],
            "description": peptide_override["description"],
        }
        # Add modifications (e.g. ptmType M3L for trimethylated lysine)
        if peptide_override.get("modifications"):
            protein_entry["modifications"] = peptide_override["modifications"]
        sequences.append({"protein": protein_entry})
    else:
        # Standard small-molecule CCD ligand
        sequences.append({
            "ligand": {
                "id": next_id,
                "ccdCodes": [ccd_code],
                "description": f"Ligand {ccd_code} from PDB {pdb_id.upper()}",
            }
        })

    return {
        "name": f"{pdb_id.lower()}_{benchmark}",
        "sequences": sequences,
        "modelSeeds": MODEL_SEEDS,
        "dialect": AF3_DIALECT,
        "version": AF3_VERSION,
    }


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def process_entry(pdb_id, protein_pdb, ligand_sdf, api_delay,
                  benchmark="casf2016", pk=None):
    result = {
        "pdb_id": pdb_id,
        "benchmark": benchmark,
        "pk": pk if pk is not None else "",
        "status": None,
        "flag_reason": "",
        "sequence_source": "",
        "num_chains": 0,
        "chain_ids_casf": "",
        "chain_ids_pdb_api": "",
        "api_reachable": False,
        "ccd_sdf_mol_name": "",
        "ccd_sdf_formula": "",
        "ccd_sdf_formula_hill": "",
        "ccd_sdf_formula_source": "",
        "ccd_sdf_mw": "",
        "ccd_code_api_all": "",
        "ccd_code_used": "",
        "ccd_resolution_method": "",
        "ligand_formula_api": "",
        "ligand_formula_api_hill": "",
        "replicated_chains": "",
        "sequence_comparison": "",
        "override_reason": "",
        "review_data": None,
        "af3_json": None,
    }

    # -- Check for manual exclusion first ----------------------------------
    if pdb_id in MANUAL_CCD_OVERRIDES and MANUAL_CCD_OVERRIDES[pdb_id] is None:
        result["status"] = "excluded"
        result["flag_reason"] = "manually excluded (see MANUAL_CCD_OVERRIDES)"
        result["ccd_resolution_method"] = "MANUAL_EXCLUDED"
        result["override_reason"] = OVERRIDE_NOTES.get(pdb_id, "")
        return result

    # Populate override_reason early for any overridden entry
    if (pdb_id in MANUAL_CCD_OVERRIDES or pdb_id in PEPTIDE_LIGAND_OVERRIDES
            or pdb_id in PPI_OVERRIDES):
        result["override_reason"] = OVERRIDE_NOTES.get(pdb_id, "")

    # -- 1. Parse benchmark PDB sequences ----------------------------------
    if not os.path.exists(protein_pdb):
        result["status"] = "error"
        result["flag_reason"] = "protein PDB file not found"
        return result

    seqres_chains = parse_seqres(protein_pdb)
    if seqres_chains:
        result["sequence_source"] = "SEQRES"
    else:
        seqres_chains = parse_atom_sequences(protein_pdb)
        result["sequence_source"] = "ATOM_fallback"

    casf_sequences = {}
    unknown_residues = set()
    for chain_id, cdata in seqres_chains.items():
        seq, unknowns = residues_to_sequence(cdata["residues"])
        casf_sequences[chain_id] = seq
        unknown_residues.update(unknowns)

    result["num_chains"] = len(casf_sequences)
    result["chain_ids_casf"] = ','.join(casf_sequences.keys())
    if unknown_residues:
        result["flag_reason"] += (
            f"non-standard residues ({','.join(sorted(unknown_residues))}); "
        )

    # -- 2. Parse SDF ligand info (RDKit primary, manual fallback) -----------
    if not os.path.exists(ligand_sdf):
        result["status"] = "error"
        result["flag_reason"] += "ligand SDF not found"
        return result

    sdf_mol_name, sdf_formula, formula_source, sdf_mw = parse_sdf_ligand_info(ligand_sdf)
    sdf_formula_hill = normalise_formula(sdf_formula) if sdf_formula else ""
    result["ccd_sdf_mol_name"] = sdf_mol_name or ""
    result["ccd_sdf_formula"] = sdf_formula or ""
    result["ccd_sdf_formula_hill"] = sdf_formula_hill
    result["ccd_sdf_formula_source"] = formula_source or ""
    result["ccd_sdf_mw"] = f"{sdf_mw:.2f}" if sdf_mw is not None else ""

    if formula_source == "rdkit":
        logging.debug(f"  -> SDF formula from RDKit: {sdf_formula_hill} (MW={sdf_mw:.2f})")

    # -- 3. Query RCSB PDB API ---------------------------------------------
    entry_data = query_pdb_api(pdb_id, delay=api_delay)
    if entry_data is None:
        result["status"] = "flagged"
        result["flag_reason"] += "PDB API unreachable or returned no data"
        result["api_reachable"] = False
        return result

    result["api_reachable"] = True
    api_chain_seqs, ligand_info = parse_api_response(entry_data)

    result["chain_ids_pdb_api"] = ','.join(sorted(api_chain_seqs.keys()))
    result["ccd_code_api_all"] = ','.join(li["ccd_id"] for li in ligand_info)
    if ligand_info:
        result["ligand_formula_api"] = ligand_info[0].get("formula", "")
        result["ligand_formula_api_hill"] = ligand_info[0].get("formula_hill", "")

    # -- 4. Resolve CCD code (or check overrides) --------------------------
    peptide_override = PEPTIDE_LIGAND_OVERRIDES.get(pdb_id)
    manual_ccd = MANUAL_CCD_OVERRIDES.get(pdb_id)
    ppi_override = PPI_OVERRIDES.get(pdb_id)

    if ppi_override:
        # PPI: the "ligand" is another protein chain; add it like a peptide
        ccd_code = None
        ccd_flag = ""
        resolution_method = "PPI_OVERRIDE"
        review_data = {}
        result["ccd_code_used"] = "PPI"
        result["ccd_resolution_method"] = resolution_method
        logging.info(
            f"  -> PPI override: adding peptide chain "
            f"'{ppi_override['sequence'][:20]}...' as protein"
        )
    elif peptide_override:
        ccd_code = None
        ccd_flag = ""
        resolution_method = "PEPTIDE_LIGAND_OVERRIDE"
        review_data = {}
        result["ccd_code_used"] = "PEPTIDE"
        result["ccd_resolution_method"] = resolution_method
        logging.info(
            f"  -> peptide ligand override: {peptide_override['sequence']}"
        )
    elif manual_ccd:
        ccd_code = manual_ccd
        ccd_flag = f"[manual override: {manual_ccd}] "
        resolution_method = "MANUAL_CCD_OVERRIDE"
        review_data = {}
        result["ccd_code_used"] = ccd_code
        result["ccd_resolution_method"] = resolution_method
    else:
        ccd_code, ccd_flag, resolution_method, review_data = resolve_ccd_code(
            pdb_id, ligand_info, sdf_mol_name, sdf_formula, sdf_mw=sdf_mw
        )
        result["ccd_resolution_method"] = resolution_method
        result["review_data"] = review_data if review_data else None

    if not peptide_override and not manual_ccd and not ppi_override:
        if ccd_flag and ccd_code is None:
            result["status"] = "flagged"
            result["flag_reason"] += ccd_flag
            return result
        if ccd_flag and ccd_code is not None:
            result["flag_reason"] += ccd_flag
        result["ccd_code_used"] = ccd_code

    # -- 5. Resolve chains and compare sequences ---------------------------
    resolved_chains, chain_notes, replicated_chains, has_unresolvable = \
        resolve_missing_chains(casf_sequences, api_chain_seqs, pdb_id)

    sequence_mismatch = has_unresolvable
    comparison_notes = list(chain_notes)

    for chain_id, casf_seq in casf_sequences.items():
        if chain_id not in api_chain_seqs:
            continue
        _, match, note = compare_sequences(casf_seq, api_chain_seqs[chain_id])
        comparison_notes.append(f"chain {chain_id}: {note}")
        if not match:
            sequence_mismatch = True

    if replicated_chains:
        result["replicated_chains"] = ','.join(sorted(replicated_chains))

    result["sequence_comparison"] = "; ".join(comparison_notes)

    if sequence_mismatch:
        result["status"] = "flagged"
        result["flag_reason"] += (
            f"sequence issues: {result['sequence_comparison']}"
        )
        return result

    # -- 6. Build AF3 JSON -------------------------------------------------
    # For PPI entries, the ppi_override dict has the same shape as
    # peptide_override (sequence, description), so we pass it through
    # the same peptide_override path to add the missing protein chain.
    effective_peptide = peptide_override or ppi_override
    result["af3_json"] = build_af3_json(
        pdb_id, resolved_chains, ccd_code,
        replicated_chains=replicated_chains,
        peptide_override=effective_peptide,
        benchmark=benchmark,
    )
    result["status"] = "verified"
    if manual_ccd:
        result["flag_reason"] = (
            f"[manual CCD override: {manual_ccd}] " + result["flag_reason"]
        )
    return result


# ---------------------------------------------------------------------------
# Manual review Markdown report
# ---------------------------------------------------------------------------

def write_review_report(results, output_path, benchmark_name=""):
    """
    Write a comprehensive audit report covering all non-trivial resolution
    decisions, not just flagged failures. This serves as documentation for
    the methods section of a paper.

    Sections:
      1. Summary statistics
      2. Excluded entries (acarbose, etc.)
      3. Manual CCD overrides (protonation states)
      4. Peptide ligand overrides (leupeptin, H3K9me3)
      5. PPI entries (protein-protein, no ligand)
      6. MW heuristic resolutions
      7. Close formula match resolutions (protonation-state auto-resolved)
      8. Remaining flagged entries (if any)
    """
    # Categorise results
    excluded = [r for r in results if r["status"] == "excluded"]
    verified = [r for r in results if r["status"] == "verified"]
    flagged = [r for r in results if r["status"] == "flagged"]

    manual_ccd = [r for r in verified
                  if r["ccd_resolution_method"] == "MANUAL_CCD_OVERRIDE"]
    peptide = [r for r in verified
               if r["ccd_resolution_method"] == "PEPTIDE_LIGAND_OVERRIDE"]
    ppi = [r for r in verified
           if r["ccd_resolution_method"] == "PPI_OVERRIDE"]
    mw_heuristic = [r for r in verified
                    if "MW_heuristic" in r.get("ccd_resolution_method", "")]
    close_match = [r for r in verified
                   if "close_match" in r.get("ccd_resolution_method", "")]
    formula_warning = [r for r in verified
                       if "formula_warning" in r.get("ccd_resolution_method", "")]
    flagged_with_review = [r for r in flagged if r.get("review_data")]

    with open(output_path, "w") as f:
        # ── Header ────────────────────────────────────────────────────────
        f.write(f"# Audit Report -- AF3 Input Generation\n\n")
        if benchmark_name:
            f.write(f"**Benchmark:** {benchmark_name}  \n")
        f.write(f"**Total entries:** {len(results)}  \n")
        f.write(f"**Verified:** {len(verified)}  \n")
        f.write(f"**Excluded:** {len(excluded)}  \n")
        f.write(f"**Flagged:** {len(flagged)}  \n\n")

        # Method breakdown
        methods = {}
        for r in verified:
            m = r["ccd_resolution_method"]
            methods[m] = methods.get(m, 0) + 1
        if methods:
            f.write("**CCD resolution methods:**\n\n")
            f.write("| Method | Count |\n|--------|-------|\n")
            for method, count in sorted(methods.items(), key=lambda x: -x[1]):
                f.write(f"| `{method}` | {count} |\n")
            f.write("\n")

        edge_case_count = (len(excluded) + len(manual_ccd) + len(peptide) +
                           len(ppi) + len(mw_heuristic) + len(close_match) +
                           len(flagged))
        f.write(
            f"**Edge cases documented below:** {edge_case_count} entries "
            f"requiring non-standard resolution.\n\n"
        )
        f.write("---\n\n")

        # ── Section 1: Excluded entries ───────────────────────────────────
        if excluded:
            f.write(f"## 1. Excluded Entries ({len(excluded)})\n\n")
            f.write(
                "These entries could not be faithfully represented in the AF3 "
                "input format and were excluded from the benchmark. This "
                "decision should be documented in the methods section.\n\n"
            )
            f.write("| PDB ID | Reason |\n|--------|--------|\n")
            for r in sorted(excluded, key=lambda x: x["pdb_id"]):
                reason = r.get("override_reason") or r.get("flag_reason", "")
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                f.write(f"| [{pdb_id}]({url}) | {reason} |\n")
            f.write("\n---\n\n")

        # ── Section 2: Manual CCD overrides (protonation states) ──────────
        if manual_ccd:
            f.write(
                f"## 2. Manual CCD Overrides -- Protonation States "
                f"({len(manual_ccd)})\n\n"
            )
            f.write(
                "These entries have SDF files where the molecular formula "
                "differs from the CCD canonical formula by 1-2 hydrogen atoms "
                "(protonation/deprotonation) or minor oxidation state "
                "differences. The parent CCD code was assigned manually. "
                "AF3 builds from CCD ideal geometry, so the protonation state "
                "in the input SDF is irrelevant for structure prediction.\n\n"
            )
            f.write(
                "| PDB ID | CCD Assigned | Rationale |\n"
                "|--------|-------------|----------|\n"
            )
            for r in sorted(manual_ccd, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                ccd = r["ccd_code_used"]
                note = OVERRIDE_NOTES.get(pdb_id, "")
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                ccd_url = f"https://www.rcsb.org/ligand/{ccd}"
                f.write(
                    f"| [{pdb_id}]({url}) | "
                    f"[{ccd}]({ccd_url}) | {note} |\n"
                )
            f.write("\n---\n\n")

        # ── Section 3: Peptide ligand overrides ───────────────────────────
        if peptide:
            f.write(f"## 3. Peptide Ligand Overrides ({len(peptide)})\n\n")
            f.write(
                "These entries have ligands that are short peptides or "
                "peptide-like molecules (e.g. leupeptin, histone H3 K9me3 "
                "peptides) classified as BIRD/PRD entries in RCSB rather than "
                "standard CCD nonpolymer ligands. They are represented as "
                "additional protein chains in the AF3 JSON, with PTM "
                "annotations where applicable.\n\n"
                "**Approximations made:**\n\n"
                "- Non-natural residues are approximated to the nearest "
                "standard amino acid\n"
                "- C-terminal aldehyde warheads (leupeptin) are approximated "
                "as standard residues\n"
                "- N-acetyl caps are dropped\n"
                "- Cyclic peptides are linearised\n\n"
            )
            f.write(
                "| PDB ID | Sequence | Modifications | Note |\n"
                "|--------|----------|---------------|------|\n"
            )
            for r in sorted(peptide, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                po = PEPTIDE_LIGAND_OVERRIDES.get(pdb_id, {})
                seq = po.get("sequence", "")
                mods = po.get("modifications", [])
                mods_str = ", ".join(
                    f"{m['ptmType']}@{m['ptmPosition']}" for m in mods
                ) if mods else "none"
                note = po.get("note", "")
                f.write(
                    f"| [{pdb_id}]({url}) | `{seq}` | {mods_str} | {note} |\n"
                )
            f.write("\n---\n\n")

        # ── Section 4: PPI entries ────────────────────────────────────────
        if ppi:
            f.write(
                f"## 4. Protein-Protein Interaction Entries ({len(ppi)})\n\n"
            )
            f.write(
                "These entries measure binding affinity between two protein "
                "chains rather than a protein-small molecule interaction. "
                "The benchmark PDB file only contains the receptor chain; "
                "the 'ligand' peptide chain is added from the RCSB FASTA "
                "as an additional protein entity in the AF3 JSON.\n\n"
            )
            f.write(
                "| PDB ID | Receptor Chains | Peptide Sequence | Note |\n"
                "|--------|----------------|-----------------|------|\n"
            )
            for r in sorted(ppi, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                chains = r.get("chain_ids_pdb_api", "")
                po = PPI_OVERRIDES.get(pdb_id, {})
                seq = po.get("sequence", "")
                note = po.get("note", "")
                f.write(
                    f"| [{pdb_id}]({url}) | {chains} | "
                    f"`{seq}` | {note} |\n"
                )
            f.write("\n---\n\n")

        # ── Section 5: MW heuristic resolutions ───────────────────────────
        if mw_heuristic:
            f.write(
                f"## 5. MW Heuristic Resolutions ({len(mw_heuristic)})\n\n"
            )
            f.write(
                "These entries had multiple non-solvent CCD candidates after "
                "filtering known additives, and no exact or close formula "
                "match. The ligand was selected by molecular weight proximity "
                "to the SDF ligand (closest MW if SDF MW available, otherwise "
                "highest MW). These should be spot-checked.\n\n"
            )
            f.write(
                "| PDB ID | CCD Selected | Method | Resolution Note |\n"
                "|--------|-------------|--------|----------------|\n"
            )
            for r in sorted(mw_heuristic, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                ccd = r["ccd_code_used"]
                method = r["ccd_resolution_method"]
                flag = r.get("flag_reason", "").strip()
                f.write(
                    f"| [{pdb_id}]({url}) | `{ccd}` | "
                    f"`{method}` | {flag} |\n"
                )
            f.write("\n---\n\n")

        # ── Section 6: Close formula match resolutions ────────────────────
        if close_match:
            f.write(
                f"## 6. Close Formula Match Resolutions ({len(close_match)})\n\n"
            )
            f.write(
                "These entries were resolved by finding a CCD candidate whose "
                "Hill formula differs from the SDF formula by 1-2 atoms "
                "(typically hydrogen, indicating protonation state "
                "differences). These were auto-resolved by the pipeline "
                "without manual intervention.\n\n"
            )
            f.write(
                "| PDB ID | CCD Selected | Method | Resolution Note |\n"
                "|--------|-------------|--------|----------------|\n"
            )
            for r in sorted(close_match, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                ccd = r["ccd_code_used"]
                method = r["ccd_resolution_method"]
                flag = r.get("flag_reason", "").strip()
                f.write(
                    f"| [{pdb_id}]({url}) | `{ccd}` | "
                    f"`{method}` | {flag} |\n"
                )
            f.write("\n---\n\n")

        # ── Section 7: Formula warning resolutions ────────────────────────
        if formula_warning:
            f.write(
                f"## 7. Formula Warning Resolutions ({len(formula_warning)})\n\n"
            )
            f.write(
                "These entries matched by SDF mol name but the Hill formula "
                "showed a small difference (1-2 atoms). Accepted as "
                "protonation/tautomer variants.\n\n"
            )
            f.write(
                "| PDB ID | CCD | Method | Note |\n"
                "|--------|-----|--------|------|\n"
            )
            for r in sorted(formula_warning, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                ccd = r["ccd_code_used"]
                method = r["ccd_resolution_method"]
                flag = r.get("flag_reason", "").strip()
                f.write(
                    f"| [{pdb_id}]({url}) | `{ccd}` | "
                    f"`{method}` | {flag} |\n"
                )
            f.write("\n---\n\n")

        # ── Section 8: Remaining flagged entries ──────────────────────────
        if flagged:
            f.write(f"## 8. Remaining Flagged Entries ({len(flagged)})\n\n")
            f.write(
                "These entries could not be automatically resolved and "
                "require manual review.\n\n"
            )
            for r in sorted(flagged, key=lambda x: x["pdb_id"]):
                pdb_id = r["pdb_id"]
                url = f"https://www.rcsb.org/structure/{pdb_id.upper()}"
                f.write(f"### {pdb_id}\n\n")
                f.write(f"**RCSB entry:** [{pdb_id.upper()}]({url})  \n")
                f.write(f"**Flag reason:** {r.get('flag_reason', '')}  \n\n")

                rd = r.get("review_data")
                if rd:
                    sdf_formula_hill = rd.get("sdf_formula_hill", "")
                    sdf_mol_name = rd.get("sdf_mol_name", "")
                    candidates = rd.get("candidates", [])
                    close = rd.get("close_candidates", [])
                    all_ccds = rd.get("all_ccd_codes", [])

                    if sdf_mol_name:
                        f.write(f"**SDF mol name:** `{sdf_mol_name}`  \n")
                    if sdf_formula_hill:
                        f.write(
                            f"**SDF formula (Hill):** `{sdf_formula_hill}`  \n"
                        )
                        sdf_mw_approx = formula_to_weight_approx(
                            sdf_formula_hill
                        )
                        f.write(f"**SDF MW (approx):** {sdf_mw_approx:.1f}  \n")
                    if all_ccds and not candidates:
                        f.write(
                            f"**All API CCDs:** "
                            f"{', '.join(f'`{c}`' for c in all_ccds)}  \n"
                        )
                    f.write("\n")

                    if candidates:
                        f.write("**Non-solvent API candidates:**\n\n")
                        f.write(
                            "| CCD | Name | Formula (Hill) | MW | RCSB | "
                            "Diff |\n|-----|------|----------------|----|----|----|\n"
                        )
                        for li in candidates:
                            ccd = li["ccd_id"]
                            fh = li.get("formula_hill", "")
                            mw = li.get("formula_weight", "")
                            mw_str = (f"{mw:.2f}" if isinstance(mw, (int, float))
                                      else str(mw or ""))
                            diffs, total = formula_atom_diff(
                                sdf_formula_hill, fh
                            )
                            diff_str = (_format_atom_diff(diffs)
                                        if total > 0 else "**exact**")
                            f.write(
                                f"| `{ccd}` | {li.get('name','')} | `{fh}` "
                                f"| {mw_str} "
                                f"| [link](https://www.rcsb.org/ligand/{ccd})"
                                f" | {diff_str} |\n"
                            )
                        f.write("\n")

                f.write("---\n\n")

        # ── Footer ────────────────────────────────────────────────────────
        if not any([excluded, manual_ccd, peptide, ppi, mw_heuristic,
                    close_match, formula_warning, flagged]):
            f.write(
                "All entries were resolved via standard automated methods "
                "(exact formula match or sole non-solvent ligand). No edge "
                "cases to document.\n"
            )

    logging.info(f"Review report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate verified AF3 input JSONs for protein-ligand benchmarks"
    )
    parser.add_argument("--coreset_dir", default="",
                        help="Path to CASF-2016 coreset directory "
                             "(mutually exclusive with --benchmark_csv)")
    parser.add_argument("--benchmark_csv", default="",
                        help="Path to a benchmark CSV file. Must have columns: "
                             "unique_id, pdb_file, sdf_file, and optionally pK.")
    parser.add_argument("--benchmark_name", default="",
                        help="Short label used in AF3 JSON name field")
    parser.add_argument("--data_root", default="",
                        help="Root directory prepended to relative paths in CSV")
    parser.add_argument("--output_json_dir", default="af_input")
    parser.add_argument("--output_csv", default="output/af3_verification.csv")
    parser.add_argument("--review_report", default="",
                        help="Path for Markdown review report (optional)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    log_path = os.path.join(
        os.path.dirname(args.output_csv) or ".", "af3_verification.log"
    )
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    log_format = "%(asctime)s %(levelname)s: %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    sh.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(sh)
    fh_log = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh_log.setLevel(logging.DEBUG)
    fh_log.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh_log)

    logging.info(f"Log file: {log_path}")
    logging.info(f"Local alignment threshold: {LOCAL_ALIGN_THRESHOLD}")
    logging.info("Formula comparison: Hill-notation normalised")
    logging.info("Formula source priority: SDF tag > RDKit (v5.1)")
    logging.info("Sequence source for AF3 JSONs: RCSB canonical (not benchmark PDB)")
    logging.info(
        f"Manual CCD overrides: "
        f"{len([v for v in MANUAL_CCD_OVERRIDES.values() if v])} entries"
    )
    logging.info(
        f"Manual exclusions: "
        f"{len([v for v in MANUAL_CCD_OVERRIDES.values() if v is None])} entries"
    )
    logging.info(
        f"Peptide ligand overrides: {len(PEPTIDE_LIGAND_OVERRIDES)} entries "
        f"({', '.join(PEPTIDE_LIGAND_OVERRIDES.keys())})"
    )
    logging.info(
        f"Conditional exclude CCDs: {len(CONDITIONAL_EXCLUDE_CCD)} "
        f"({', '.join(sorted(CONDITIONAL_EXCLUDE_CCD))})"
    )
    logging.info(f"Known additive CCDs: {len(KNOWN_ADDITIVE_CCD)}")
    logging.info(
        f"PPI overrides (no ligand): {len(PPI_OVERRIDES)} entries "
        f"({', '.join(sorted(PPI_OVERRIDES))})"
    )

    os.makedirs(args.output_json_dir, exist_ok=True)

    # -- Build entry list --------------------------------------------------
    if args.benchmark_csv and args.coreset_dir:
        parser.error("--coreset_dir and --benchmark_csv are mutually exclusive")

    benchmark_name = args.benchmark_name
    if not benchmark_name:
        if args.benchmark_csv:
            base = os.path.splitext(os.path.basename(args.benchmark_csv))[0]
            benchmark_name = re.sub(r'[_-](test|val|train|eval)$', '', base,
                                    flags=re.IGNORECASE)
            benchmark_name = benchmark_name.lower().replace("-", "").replace("_", "")
        else:
            benchmark_name = "casf2016"

    entries = []

    if args.benchmark_csv:
        import csv as _csv
        data_root = args.data_root.rstrip("/")
        with open(args.benchmark_csv, newline="") as csvfile:
            reader = _csv.DictReader(csvfile)
            for row in reader:
                pdb_id = row["unique_id"].strip().lower()
                pdb_file = row["pdb_file"].strip()
                sdf_file = row["sdf_file"].strip()
                if data_root:
                    if not os.path.isabs(pdb_file):
                        pdb_file = os.path.join(data_root, pdb_file)
                    if not os.path.isabs(sdf_file):
                        sdf_file = os.path.join(data_root, sdf_file)
                pk = row.get("pK", row.get("pk", "")).strip()
                entries.append({
                    "pdb_id": pdb_id,
                    "protein_pdb": pdb_file,
                    "ligand_sdf": sdf_file,
                    "pk": pk,
                })
    else:
        coreset_dir = args.coreset_dir or "data/CASF-2016/coreset"
        for pdb_id in sorted(os.listdir(coreset_dir)):
            entry_dir = os.path.join(coreset_dir, pdb_id)
            if not os.path.isdir(entry_dir):
                continue
            entries.append({
                "pdb_id": pdb_id,
                "protein_pdb": os.path.join(entry_dir, f"{pdb_id}_protein.pdb"),
                "ligand_sdf": os.path.join(entry_dir, f"{pdb_id}_ligand.sdf"),
                "pk": "",
            })

    if args.limit > 0:
        entries = entries[:args.limit]

    print(f"Benchmark    : {benchmark_name}")
    print(f"Mode         : {'CSV (' + args.benchmark_csv + ')' if args.benchmark_csv else 'directory scan'}")
    print(f"Entries      : {len(entries)}")
    print(f"Output JSONs -> {args.output_json_dir}/")
    print(f"Audit CSV   -> {args.output_csv}")
    print("=" * 70)

    results = []
    counts = {}

    for i, entry in enumerate(entries, 1):
        pdb_id = entry["pdb_id"]
        logging.info(f"[{i}/{len(entries)}] {pdb_id}")

        result = process_entry(
            pdb_id=pdb_id,
            protein_pdb=entry["protein_pdb"],
            ligand_sdf=entry["ligand_sdf"],
            api_delay=args.delay,
            benchmark=benchmark_name,
            pk=entry.get("pk") or None,
        )
        results.append(result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1

        if result["status"] == "verified":
            json_path = os.path.join(args.output_json_dir, f"{pdb_id}.json")
            with open(json_path, "w") as f:
                json.dump(result["af3_json"], f, indent=2)
            method = result["ccd_resolution_method"]
            rep = (f" replicated={result['replicated_chains']}"
                   if result["replicated_chains"] else "")
            formula_src = (f" formula_src={result['ccd_sdf_formula_source']}"
                           if result.get("ccd_sdf_formula_source") else "")
            logging.info(
                f"  -> verified [{method}]{rep}{formula_src} -> {json_path}"
            )
        elif result["status"] == "excluded":
            logging.info(f"  -> excluded: {result['flag_reason']}")
        else:
            logging.warning(
                f"  X {result['status']}: {result['flag_reason'][:120]}"
            )

        if args.verbose and result.get("sequence_comparison"):
            for line in result["sequence_comparison"].split(";"):
                if line.strip():
                    logging.debug(f"    {line.strip()}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entries processed: {len(entries)}")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    verified = [r for r in results if r["status"] == "verified"]
    if verified:
        methods = {}
        for r in verified:
            m = r["ccd_resolution_method"]
            methods[m] = methods.get(m, 0) + 1
        print(f"\nCCD resolution methods ({len(verified)} verified):")
        for method, count in sorted(methods.items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")

        # Formula source stats
        formula_sources = {}
        for r in verified:
            src = r.get("ccd_sdf_formula_source", "") or "none"
            formula_sources[src] = formula_sources.get(src, 0) + 1
        print(f"\nSDF formula sources ({len(verified)} verified):")
        for src, count in sorted(formula_sources.items(), key=lambda x: -x[1]):
            print(f"  {src}: {count}")

    excluded = [r for r in results if r["status"] == "excluded"]
    if excluded:
        print(f"\nExcluded ({len(excluded)}):")
        for r in excluded:
            reason = (r['override_reason'][:100]
                      if r['override_reason'] else r['flag_reason'][:100])
            print(f"  {r['pdb_id']}: {reason}")

    peptide_entries = [r for r in results
                       if r["ccd_resolution_method"] == "PEPTIDE_LIGAND_OVERRIDE"]
    if peptide_entries:
        print(f"\nPeptide ligand entries: {[r['pdb_id'] for r in peptide_entries]}")

    ppi_entries = [r for r in results
                   if r["ccd_resolution_method"] == "PPI_OVERRIDE"]
    if ppi_entries:
        print(f"PPI entries (no ligand): {[r['pdb_id'] for r in ppi_entries]}")

    flagged = [r for r in results if r["status"] == "flagged"]
    if flagged:
        print(f"\nFlagged ({len(flagged)}):")
        for r in flagged:
            print(f"  {r['pdb_id']}: {r['flag_reason'][:120]}")

    # Write CSV
    csv_fields = [
        "pdb_id", "benchmark", "pk",
        "status", "flag_reason", "override_reason",
        "sequence_source",
        "num_chains", "chain_ids_casf", "chain_ids_pdb_api",
        "api_reachable",
        "ccd_sdf_mol_name", "ccd_sdf_formula", "ccd_sdf_formula_hill",
        "ccd_sdf_formula_source", "ccd_sdf_mw",
        "ccd_code_api_all", "ccd_code_used", "ccd_resolution_method",
        "ligand_formula_api", "ligand_formula_api_hill",
        "replicated_chains", "sequence_comparison",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAudit CSV -> {args.output_csv}")

    if args.review_report:
        os.makedirs(os.path.dirname(args.review_report) or ".", exist_ok=True)
        write_review_report(results, args.review_report, benchmark_name)
        print(f"Review report -> {args.review_report}")

    print(f"AF3 JSONs -> {args.output_json_dir}/ ({len(verified)} files)")


if __name__ == "__main__":
    main()