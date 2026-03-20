#!/usr/bin/env python3
"""
compute_rmsd.py — Calculate protein Cα RMSD and ligand heavy-atom RMSD
between AF3-predicted structures and experimental crystal structures.

Handles:
  - Chain matching by sequence similarity (not chain letter)
  - Truncated crystal structures (PDBbind binding-site regions)
  - Both PDB and mmCIF file formats
  - Proper Kabsch superposition with ligand transform
  - Peptide/PPI entries (protein-only RMSD, no ligand RMSD)

Dependencies:
    pip install gemmi numpy scipy

Usage:
    python compute_rmsd.py \
        --af3_dir /path/to/af_processed \
        --data_root /path/to/DAT675-Project \
        --benchmarks casf2016 0ligandbias oodtest \
        --output_csv rmsd_results.csv \
        --verbose

Author: Generated for DAT675 Project AF3 evaluation
"""

import argparse
import csv
import json
import logging
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

# Try importing gemmi (preferred for CIF/PDB parsing)
try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False
    warnings.warn("gemmi not available; falling back to manual PDB parser. "
                  "Install gemmi for better CIF support: pip install gemmi")


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O',
    'MSE': 'M', 'HYP': 'P', 'SEP': 'S', 'TPO': 'T', 'CSO': 'C',
    'PTR': 'Y', 'MLY': 'K', 'KCX': 'K', 'CSS': 'C', 'CSD': 'C',
    'OIC': 'P', 'AIB': 'A', 'NME': 'G',
}

# Entries that have peptide ligands or PPI (no small-molecule ligand)
PEPTIDE_LIGAND_ENTRIES = {
    "1a30", "3bv9", "3uri",           # CASF-2016 peptide ligands
    "1jrs", "1tl9", "1pop",           # 0LigandBias leupeptin
    "2l3r", "2l75", "4yhp",           # 0LigandBias H3K9me3 peptides
}
PPI_ENTRIES = {"1g1e", "1s5q"}
NO_SDF_ENTRIES = PEPTIDE_LIGAND_ENTRIES | PPI_ENTRIES

# Excluded entries (from generate_af3_inputs.py — acarbose etc.)
EXCLUDED_ENTRIES = {
    "1gah", "1lf9", "2zq0", "3jyr", "3jzj", "1k1y", "2qmj",
}


# ═══════════════════════════════════════════════════════════════════════════
# Structure parsing
# ═══════════════════════════════════════════════════════════════════════════

class ProteinChain:
    """Represents a protein chain with residue-level data."""
    def __init__(self, chain_id):
        self.chain_id = chain_id
        self.residues = []  # list of (resnum, resname_3, x_ca, y_ca, z_ca)
        self.sequence = ""  # one-letter code

    def build_sequence(self):
        seq = []
        for _, resname, _, _, _ in self.residues:
            aa = THREE_TO_ONE.get(resname, 'X')
            seq.append(aa)
        self.sequence = ''.join(seq)
        return self.sequence

    def get_ca_coords(self):
        """Return (N,3) array of Cα coordinates."""
        coords = [(x, y, z) for _, _, x, y, z in self.residues]
        return np.array(coords) if coords else np.empty((0, 3))

    def get_resnums(self):
        return [r[0] for r in self.residues]


class LigandAtom:
    """Single atom in a ligand."""
    def __init__(self, name, element, x, y, z):
        self.name = name.strip()
        self.element = element.strip()
        self.x, self.y, self.z = x, y, z

    def is_heavy(self):
        return self.element.upper() not in ('H', 'D')

    def coords(self):
        return np.array([self.x, self.y, self.z])


class Ligand:
    """Represents a ligand with atoms."""
    def __init__(self, resname, chain_id=""):
        self.resname = resname
        self.chain_id = chain_id
        self.atoms = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def heavy_atoms(self):
        return [a for a in self.atoms if a.is_heavy()]

    def heavy_atom_coords(self):
        ha = self.heavy_atoms()
        if not ha:
            return np.empty((0, 3))
        return np.array([a.coords() for a in ha])

    def heavy_atom_names(self):
        return [a.name for a in self.heavy_atoms()]


def parse_pdb_structure(pdb_path):
    """
    Parse a PDB file and extract protein chains (Cα atoms) and ligands.
    Returns (dict of chain_id -> ProteinChain, list of Ligand).
    """
    chains = {}
    ligands = {}
    seen_ca = {}  # (chain_id, resnum) -> True, to avoid duplicates

    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()

            if record == "ATOM":
                atom_name = line[12:16].strip()
                chain_id = line[21].strip()
                resname = line[17:20].strip()
                try:
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                # Only collect Cα atoms for protein chains
                if atom_name == "CA" and resname in THREE_TO_ONE:
                    key = (chain_id, resnum)
                    if key not in seen_ca:
                        seen_ca[key] = True
                        if chain_id not in chains:
                            chains[chain_id] = ProteinChain(chain_id)
                        chains[chain_id].residues.append(
                            (resnum, resname, x, y, z))

            elif record == "HETATM":
                atom_name = line[12:16].strip()
                chain_id = line[21].strip()
                resname = line[17:20].strip()
                try:
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                # Guess element from columns 76-78 or atom name
                element = line[76:78].strip() if len(line) > 77 else ""
                if not element:
                    # Guess from atom name
                    element = ''.join(c for c in atom_name if c.isalpha())
                    if len(element) > 1 and element[1].islower():
                        element = element[:2]
                    else:
                        element = element[0] if element else "C"

                lig_key = (chain_id, resname, resnum)
                if lig_key not in ligands:
                    ligands[lig_key] = Ligand(resname, chain_id)
                ligands[lig_key].add_atom(
                    LigandAtom(atom_name, element, x, y, z))

    # Build sequences
    for chain in chains.values():
        chain.build_sequence()

    return chains, list(ligands.values())


def parse_pdb_structure_gemmi(pdb_path):
    """
    Parse PDB/CIF using gemmi. Returns (chains_dict, ligands_list).
    """
    st = gemmi.read_structure(pdb_path)
    if not st or not st[0]:
        return {}, []

    model = st[0]
    chains = {}
    ligands = {}
    seen_ca = {}

    for chain in model:
        chain_id = chain.name

        for residue in chain:
            is_polymer = residue.entity_type == gemmi.EntityType.Polymer
            is_standard = residue.name in THREE_TO_ONE

            if is_polymer or is_standard:
                # Protein residue — look for CA
                for atom in residue:
                    if atom.name == "CA":
                        key = (chain_id, residue.seqid.num)
                        if key not in seen_ca:
                            seen_ca[key] = True
                            if chain_id not in chains:
                                chains[chain_id] = ProteinChain(chain_id)
                            chains[chain_id].residues.append((
                                residue.seqid.num, residue.name,
                                atom.pos.x, atom.pos.y, atom.pos.z
                            ))
            else:
                # Non-polymer = potential ligand
                lig_key = (chain_id, residue.name, residue.seqid.num)
                if lig_key not in ligands:
                    ligands[lig_key] = Ligand(residue.name, chain_id)
                for atom in residue:
                    ligands[lig_key].add_atom(
                        LigandAtom(atom.name, atom.element.name,
                                   atom.pos.x, atom.pos.y, atom.pos.z)
                    )

    for chain in chains.values():
        chain.build_sequence()

    return chains, list(ligands.values())


def parse_structure(path):
    """Auto-select parser based on file extension and availability."""
    if HAS_GEMMI:
        try:
            return parse_pdb_structure_gemmi(path)
        except Exception as e:
            logging.debug(f"gemmi parse failed for {path}: {e}, falling back")
    return parse_pdb_structure(path)


def parse_sdf_ligand(sdf_path):
    """
    Parse an SDF file to extract atom names, elements, and coordinates.
    Returns a Ligand object.
    """
    ligand = Ligand("LIG")
    try:
        with open(sdf_path) as f:
            lines = f.readlines()
    except Exception:
        return ligand

    if len(lines) < 4:
        return ligand

    # Parse counts line (line 4, 0-indexed line 3)
    try:
        counts_line = lines[3]
        n_atoms = int(counts_line[0:3].strip())
        n_bonds = int(counts_line[3:6].strip())
    except (ValueError, IndexError):
        return ligand

    # Parse atom block
    for i in range(4, min(4 + n_atoms, len(lines))):
        line = lines[i]
        try:
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            element = line[31:34].strip()
            # Atom name: use element + index
            atom_name = f"{element}{i - 3}"
        except (ValueError, IndexError):
            continue
        ligand.add_atom(LigandAtom(atom_name, element, x, y, z))

    return ligand


def parse_sdf_ligand_gemmi(sdf_path):
    """Parse SDF using gemmi if available. Falls back to manual parser."""
    if not HAS_GEMMI:
        return parse_sdf_ligand(sdf_path)
    try:
        # gemmi can read SDF/MOL files
        # But for SDF, manual parsing is often more reliable
        # Let's try gemmi's approach first
        return parse_sdf_ligand(sdf_path)
    except Exception:
        return parse_sdf_ligand(sdf_path)


# ═══════════════════════════════════════════════════════════════════════════
# Sequence alignment for chain matching
# ═══════════════════════════════════════════════════════════════════════════

def needleman_wunsch(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Simple Needleman-Wunsch global alignment.
    Returns (aligned_seq1, aligned_seq2, score).
    """
    n, m = len(seq1), len(seq2)
    # Score matrix
    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                dp[i - 1][j - 1] + s,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )

    # Traceback
    aligned1, aligned2 = [], []
    i, j = n, m
    while i > 0 and j > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch
        if dp[i][j] == dp[i - 1][j - 1] + s:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + gap:
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            j -= 1
    while i > 0:
        aligned1.append(seq1[i - 1])
        aligned2.append('-')
        i -= 1
    while j > 0:
        aligned1.append('-')
        aligned2.append(seq2[j - 1])
        j -= 1

    return ''.join(reversed(aligned1)), ''.join(reversed(aligned2)), dp[n][m]


def smith_waterman(seq1, seq2, match=2, mismatch=-1, gap=-1):
    """
    Smith-Waterman local alignment.
    Returns (aligned_seq1, aligned_seq2, score, start_i, start_j).
    Better for finding the best local match when one sequence is a
    truncated version of the other.
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=float)
    max_score = 0
    max_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(
                0,
                dp[i - 1][j - 1] + s,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
            if dp[i][j] > max_score:
                max_score = dp[i][j]
                max_pos = (i, j)

    # Traceback from max_pos
    aligned1, aligned2 = [], []
    i, j = max_pos
    while i > 0 and j > 0 and dp[i][j] > 0:
        s = match if seq1[i - 1] == seq2[j - 1] else mismatch
        if dp[i][j] == dp[i - 1][j - 1] + s:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + gap:
            aligned1.append(seq1[i - 1])
            aligned2.append('-')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j - 1])
            j -= 1

    return (''.join(reversed(aligned1)), ''.join(reversed(aligned2)),
            max_score, i, j)


def sequence_identity(seq1, seq2):
    """Compute sequence identity between two sequences (local alignment)."""
    if not seq1 or not seq2:
        return 0.0
    a1, a2, score, _, _ = smith_waterman(seq1, seq2)
    if not a1:
        return 0.0
    matches = sum(1 for c1, c2 in zip(a1, a2) if c1 == c2 and c1 != '-')
    return matches / max(len(a1.replace('-', '')), 1)


# ═══════════════════════════════════════════════════════════════════════════
# Chain matching
# ═══════════════════════════════════════════════════════════════════════════

def match_chains(af3_chains, xtal_chains, min_identity=0.5):
    """
    Match AF3 chains to crystal structure chains by sequence similarity.

    For each crystal chain, find the AF3 chain with the highest
    sequence identity. Uses Smith-Waterman local alignment to handle
    truncated crystal structures.

    Returns list of (xtal_chain_id, af3_chain_id, identity_score).
    """
    matches = []
    used_af3 = set()

    for xtal_id, xtal_chain in xtal_chains.items():
        best_af3_id = None
        best_identity = 0.0

        for af3_id, af3_chain in af3_chains.items():
            if af3_id in used_af3:
                continue
            identity = sequence_identity(xtal_chain.sequence, af3_chain.sequence)
            if identity > best_identity:
                best_identity = identity
                best_af3_id = af3_id

        if best_af3_id and best_identity >= min_identity:
            matches.append((xtal_id, best_af3_id, best_identity))
            used_af3.add(best_af3_id)
        else:
            logging.debug(
                f"  No match for xtal chain {xtal_id} "
                f"(best identity: {best_identity:.2f})"
            )

    return matches


# ═══════════════════════════════════════════════════════════════════════════
# Residue matching within aligned chains
# ═══════════════════════════════════════════════════════════════════════════

def get_matched_ca_coords(af3_chain, xtal_chain):
    """
    Get matched Cα coordinates between AF3 and crystal chains.

    Uses sequence alignment to find corresponding residues, then
    extracts the Cα coordinates for residues present in both structures.

    Returns (af3_coords, xtal_coords) as (N,3) arrays with matched rows.
    """
    af3_seq = af3_chain.sequence
    xtal_seq = xtal_chain.sequence

    if not af3_seq or not xtal_seq:
        return np.empty((0, 3)), np.empty((0, 3))

    # Use local alignment since xtal may be truncated
    a1, a2, score, start_i, start_j = smith_waterman(
        xtal_seq, af3_seq, match=2, mismatch=-1, gap=-2
    )

    # Map aligned positions back to residue indices
    af3_coords_list = []
    xtal_coords_list = []

    xtal_idx = start_i  # index into xtal_chain.residues
    af3_idx = start_j   # index into af3_chain.residues

    for c1, c2 in zip(a1, a2):
        if c1 != '-' and c2 != '-':
            # Both have a residue at this aligned position
            if c1 == c2:  # Only use matching positions
                if (xtal_idx < len(xtal_chain.residues) and
                        af3_idx < len(af3_chain.residues)):
                    xr = xtal_chain.residues[xtal_idx]
                    ar = af3_chain.residues[af3_idx]
                    xtal_coords_list.append([xr[2], xr[3], xr[4]])
                    af3_coords_list.append([ar[2], ar[3], ar[4]])
            if c1 != '-':
                xtal_idx += 1
            if c2 != '-':
                af3_idx += 1
        elif c1 != '-':
            xtal_idx += 1
        elif c2 != '-':
            af3_idx += 1

    if not af3_coords_list:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.array(af3_coords_list), np.array(xtal_coords_list)


# ═══════════════════════════════════════════════════════════════════════════
# Kabsch algorithm
# ═══════════════════════════════════════════════════════════════════════════

def kabsch_rmsd(P, Q):
    """
    Compute RMSD after optimal superposition using the Kabsch algorithm.

    P: (N,3) array — mobile coordinates (AF3)
    Q: (N,3) array — reference coordinates (crystal)

    Returns (rmsd, rotation_matrix, translation_to_centroid_Q,
             centroid_P, centroid_Q)
    """
    assert P.shape == Q.shape
    n = P.shape[0]
    if n == 0:
        return float('inf'), np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3)

    # Center both
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute cross-covariance matrix
    H = p.T @ q

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.eye(3)
    sign_matrix[2, 2] = np.sign(d)

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply rotation and compute RMSD
    p_rotated = p @ R.T
    diff = p_rotated - q
    rmsd = np.sqrt((diff ** 2).sum() / n)

    return rmsd, R, centroid_P, centroid_Q


def apply_transform(coords, R, centroid_P, centroid_Q):
    """
    Apply the Kabsch transformation to a set of coordinates.
    coords: (N,3) — coordinates to transform (in the AF3 frame)
    Returns transformed coordinates aligned to the crystal frame.
    """
    centered = coords - centroid_P
    rotated = centered @ R.T
    return rotated + centroid_Q


# ═══════════════════════════════════════════════════════════════════════════
# Ligand RMSD
# ═══════════════════════════════════════════════════════════════════════════

def match_ligand_atoms(af3_ligand, xtal_ligand):
    """
    Match ligand heavy atoms between AF3 and crystal structures.

    Strategy:
    1. Try exact atom name matching
    2. If that fails, try element-based matching (ordered)

    Returns (af3_coords, xtal_coords) as matched (N,3) arrays.
    """
    af3_heavy = af3_ligand.heavy_atoms()
    xtal_heavy = xtal_ligand.heavy_atoms()

    if not af3_heavy or not xtal_heavy:
        return np.empty((0, 3)), np.empty((0, 3))

    # Strategy 1: Match by atom name
    af3_by_name = {a.name: a for a in af3_heavy}
    xtal_by_name = {a.name: a for a in xtal_heavy}

    common_names = set(af3_by_name.keys()) & set(xtal_by_name.keys())

    if len(common_names) >= min(len(af3_heavy), len(xtal_heavy)) * 0.8:
        # Good name matching — use it
        af3_coords = np.array([af3_by_name[n].coords() for n in sorted(common_names)])
        xtal_coords = np.array([xtal_by_name[n].coords() for n in sorted(common_names)])
        return af3_coords, xtal_coords

    # Strategy 2: Match by element, preserving order within each element
    af3_by_elem = defaultdict(list)
    xtal_by_elem = defaultdict(list)
    for a in af3_heavy:
        af3_by_elem[a.element.upper()].append(a)
    for a in xtal_heavy:
        xtal_by_elem[a.element.upper()].append(a)

    af3_matched, xtal_matched = [], []
    for elem in sorted(set(af3_by_elem.keys()) & set(xtal_by_elem.keys())):
        n = min(len(af3_by_elem[elem]), len(xtal_by_elem[elem]))
        for i in range(n):
            af3_matched.append(af3_by_elem[elem][i].coords())
            xtal_matched.append(xtal_by_elem[elem][i].coords())

    if not af3_matched:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.array(af3_matched), np.array(xtal_matched)


def compute_ligand_rmsd(af3_ligand_coords, xtal_ligand_coords,
                        R, centroid_P, centroid_Q):
    """
    Compute ligand RMSD after applying protein superposition transform.

    The AF3 ligand coordinates are transformed using the rotation and
    translation from the protein Kabsch superposition, then RMSD is
    computed against the crystal ligand coordinates.

    This measures how well AF3 placed the ligand in the binding pocket.
    """
    if af3_ligand_coords.shape[0] == 0 or xtal_ligand_coords.shape[0] == 0:
        return float('inf')

    n = min(af3_ligand_coords.shape[0], xtal_ligand_coords.shape[0])
    af3_transformed = apply_transform(
        af3_ligand_coords[:n], R, centroid_P, centroid_Q
    )
    diff = af3_transformed - xtal_ligand_coords[:n]
    return np.sqrt((diff ** 2).sum() / n)


# ═══════════════════════════════════════════════════════════════════════════
# Finding the correct ligand in crystal structure
# ═══════════════════════════════════════════════════════════════════════════

# Solvents / additives to skip when identifying the benchmark ligand
SOLVENT_RESNAMES = {
    "HOH", "WAT", "DOD", "GOL", "EDO", "PEG", "DMS", "SO4", "PO4",
    "CL", "NA", "MG", "ZN", "CA", "MN", "FE", "CO", "NI", "CU",
    "IOD", "BR", "FE2", "CU1", "K", "ACE", "ACT", "TRS", "EPE",
    "MES", "PIP", "IMD", "TAR", "MPD", "IPA", "EOH", "MOH",
    "EGL", "PGE", "PG4", "1PE", "DTT", "BME", "NAG", "MAN",
    "FUC", "GAL", "GLC", "SIA", "BGC", "BMA",
}


def find_ligand_in_structure(ligands, target_ccd=None):
    """
    Identify the benchmark ligand from a list of ligand residues.

    If target_ccd is given, prefer that CCD code.
    Otherwise, pick the non-solvent ligand with the most heavy atoms.
    """
    if not ligands:
        return None

    if target_ccd:
        matches = [lig for lig in ligands if lig.resname == target_ccd]
        if matches:
            # Return the one with most atoms
            return max(matches, key=lambda l: len(l.heavy_atoms()))

    # Filter out solvents/ions
    candidates = [lig for lig in ligands if lig.resname not in SOLVENT_RESNAMES]
    if not candidates:
        return None

    # Return the largest non-solvent ligand
    return max(candidates, key=lambda l: len(l.heavy_atoms()))


# ═══════════════════════════════════════════════════════════════════════════
# Main RMSD computation for one entry
# ═══════════════════════════════════════════════════════════════════════════

def compute_entry_rmsd(af3_protein_path, af3_ligand_path,
                       xtal_protein_path, xtal_ligand_path,
                       pdb_id, ccd_code=None):
    """
    Compute protein Cα RMSD and ligand RMSD for a single entry.

    Returns dict with:
        protein_rmsd, ligand_rmsd, n_ca_matched, n_ligand_atoms_matched,
        chain_matches, warnings
    """
    result = {
        "protein_rmsd": None,
        "ligand_rmsd": None,
        "n_ca_matched": 0,
        "n_ligand_atoms_matched": 0,
        "n_af3_chains": 0,
        "n_xtal_chains": 0,
        "chain_matches": "",
        "warnings": [],
    }

    # ── Load structures ──
    if not os.path.exists(af3_protein_path):
        result["warnings"].append("AF3 protein file not found")
        return result

    if not os.path.exists(xtal_protein_path):
        result["warnings"].append("Crystal protein file not found")
        return result

    af3_chains, af3_ligands = parse_structure(af3_protein_path)
    xtal_chains, xtal_ligands = parse_structure(xtal_protein_path)

    result["n_af3_chains"] = len(af3_chains)
    result["n_xtal_chains"] = len(xtal_chains)

    if not af3_chains:
        result["warnings"].append("No protein chains in AF3 structure")
        return result
    if not xtal_chains:
        result["warnings"].append("No protein chains in crystal structure")
        return result

    # ── Match chains by sequence ──
    chain_matches = match_chains(af3_chains, xtal_chains, min_identity=0.5)
    if not chain_matches:
        # Try with lower threshold
        chain_matches = match_chains(af3_chains, xtal_chains, min_identity=0.3)

    if not chain_matches:
        result["warnings"].append("No chain matches found by sequence")
        return result

    result["chain_matches"] = "; ".join(
        f"xtal:{x}->af3:{a} ({ident:.2f})"
        for x, a, ident in chain_matches
    )

    # ── Collect matched Cα coordinates across all chain pairs ──
    all_af3_ca = []
    all_xtal_ca = []

    for xtal_cid, af3_cid, identity in chain_matches:
        af3_ca, xtal_ca = get_matched_ca_coords(
            af3_chains[af3_cid], xtal_chains[xtal_cid]
        )
        if af3_ca.shape[0] > 0:
            all_af3_ca.append(af3_ca)
            all_xtal_ca.append(xtal_ca)
            logging.debug(
                f"  Chain {xtal_cid}->{af3_cid}: "
                f"{af3_ca.shape[0]} matched Cα atoms"
            )

    if not all_af3_ca:
        result["warnings"].append("No matched Cα atoms found")
        return result

    af3_ca_all = np.vstack(all_af3_ca)
    xtal_ca_all = np.vstack(all_xtal_ca)
    result["n_ca_matched"] = af3_ca_all.shape[0]

    if af3_ca_all.shape[0] < 3:
        result["warnings"].append(
            f"Too few Cα atoms matched ({af3_ca_all.shape[0]})")
        return result

    # ── Protein superposition (Kabsch) ──
    protein_rmsd, R, centroid_P, centroid_Q = kabsch_rmsd(af3_ca_all, xtal_ca_all)
    result["protein_rmsd"] = protein_rmsd

    # ── Ligand RMSD ──
    is_peptide = pdb_id.lower() in NO_SDF_ENTRIES
    if is_peptide:
        result["warnings"].append("peptide/PPI entry — no ligand RMSD")
        return result

    # Load AF3 ligand
    af3_ligand = None
    if af3_ligand_path and os.path.exists(af3_ligand_path):
        af3_ligand_from_sdf = parse_sdf_ligand(af3_ligand_path)
        if af3_ligand_from_sdf.heavy_atoms():
            af3_ligand = af3_ligand_from_sdf
        else:
            # Try finding ligand in the protein PDB (AF3 sometimes includes it)
            af3_ligand = find_ligand_in_structure(af3_ligands, ccd_code)
    else:
        # No SDF — look in the AF3 protein structure
        af3_ligand = find_ligand_in_structure(af3_ligands, ccd_code)

    if af3_ligand is None or not af3_ligand.heavy_atoms():
        result["warnings"].append("AF3 ligand not found or empty")
        return result

    # Load crystal ligand
    xtal_ligand = None
    if xtal_ligand_path and os.path.exists(xtal_ligand_path):
        xtal_ligand_from_sdf = parse_sdf_ligand(xtal_ligand_path)
        if xtal_ligand_from_sdf.heavy_atoms():
            xtal_ligand = xtal_ligand_from_sdf
        else:
            xtal_ligand = find_ligand_in_structure(xtal_ligands, ccd_code)
    else:
        xtal_ligand = find_ligand_in_structure(xtal_ligands, ccd_code)

    if xtal_ligand is None or not xtal_ligand.heavy_atoms():
        result["warnings"].append("Crystal ligand not found or empty")
        return result

    # Match ligand atoms
    af3_lig_coords, xtal_lig_coords = match_ligand_atoms(af3_ligand, xtal_ligand)

    if af3_lig_coords.shape[0] == 0:
        result["warnings"].append("No ligand atoms matched")
        return result

    result["n_ligand_atoms_matched"] = af3_lig_coords.shape[0]

    # Compute ligand RMSD using protein superposition transform
    ligand_rmsd = compute_ligand_rmsd(
        af3_lig_coords, xtal_lig_coords, R, centroid_P, centroid_Q
    )
    result["ligand_rmsd"] = ligand_rmsd

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════

def load_benchmark_entries(benchmark_csv, af3_csv, data_root, af3_root, benchmark):
    """
    Load entries from benchmark and AF3 metadata CSVs.

    Returns list of dicts with:
        pdb_id, pk, xtal_protein, xtal_ligand_sdf,
        af3_protein, af3_ligand_sdf, ccd_code
    """
    # Load CCD codes from AF3 verification/test CSV
    ccd_map = {}
    if af3_csv and os.path.exists(af3_csv):
        with open(af3_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("unique_id", "").strip().lower()
                ccd = row.get("ccd_code_used", "").strip()
                if not ccd:
                    # Try from sdf_file path
                    pass
                if uid:
                    ccd_map[uid] = ccd

    entries = []

    with open(benchmark_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row["unique_id"].strip().lower()

            if pdb_id in EXCLUDED_ENTRIES:
                continue

            # Crystal structure paths (resolve relative to data_root)
            xtal_protein = row.get("pdb_file", "").strip()
            xtal_ligand = row.get("sdf_file", "").strip()

            if data_root:
                if not os.path.isabs(xtal_protein):
                    xtal_protein = os.path.join(data_root, xtal_protein)
                if not os.path.isabs(xtal_ligand):
                    xtal_ligand = os.path.join(data_root, xtal_ligand)

            # AF3 structure paths
            af3_dir = os.path.join(af3_root, benchmark, pdb_id)
            af3_protein = os.path.join(af3_dir, f"{pdb_id}_protein.pdb")
            af3_ligand = os.path.join(af3_dir, f"{pdb_id}_ligand.sdf")

            entries.append({
                "pdb_id": pdb_id,
                "pk": row.get("pK", row.get("pk", "")),
                "xtal_protein": xtal_protein,
                "xtal_ligand_sdf": xtal_ligand,
                "af3_protein": af3_protein,
                "af3_ligand_sdf": af3_ligand,
                "ccd_code": ccd_map.get(pdb_id, ""),
                "benchmark": benchmark,
            })

    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_statistics(values, label=""):
    """Compute summary statistics for a list of values."""
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None,
                "pct_below_2A": None, "pct_below_5A": None}
    arr = np.array(values)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "pct_below_2A": float(np.sum(arr < 2.0) / len(arr) * 100),
        "pct_below_5A": float(np.sum(arr < 5.0) / len(arr) * 100),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute protein Cα RMSD and ligand RMSD for AF3 predictions"
    )
    parser.add_argument(
        "--af3_dir",
        default="/mimer/NOBACKUP/groups/naiss2023-6-290/nils/DAT675-Project/data/af_processed",
        help="Root directory with AF3 processed structures",
    )
    parser.add_argument(
        "--data_root",
        default="/mimer/NOBACKUP/groups/naiss2023-6-290/nils/DAT675-Project",
        help="Root directory for resolving relative paths in CSV",
    )
    parser.add_argument(
        "--eval_dir",
        default="/mimer/NOBACKUP/groups/naiss2023-6-290/nils/DAT675-Project/evaluate",
        help="Directory containing benchmark evaluation CSVs",
    )
    parser.add_argument(
        "--benchmarks", nargs="*",
        default=["casf2016", "0ligandbias", "oodtest"],
        help="Benchmark names to process",
    )
    parser.add_argument(
        "--output_csv", default="rmsd_results.csv",
        help="Output CSV with per-entry RMSD values",
    )
    parser.add_argument(
        "--output_stats", default="rmsd_statistics.csv",
        help="Output CSV with per-benchmark summary statistics",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit entries per benchmark (0 = all, for debugging)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if HAS_GEMMI:
        logging.info(f"Using gemmi {gemmi.__version__} for structure parsing")
    else:
        logging.info("Using manual PDB parser (install gemmi for CIF support)")

    # ── Benchmark configuration ──
    # Maps benchmark name -> (benchmark_csv, af3_csv, benchmark_subdir_in_af3)
    benchmark_config = {
        "casf2016": {
            "benchmark_csv": os.path.join(
                args.eval_dir, "casf-2016", "casf2016_test.csv"),
            "af3_csv": os.path.join(
                args.af3_dir, "..", "af_processed",
                "casf2016_af3_test.csv") if False else
                os.path.join(args.eval_dir, "casf-2016", "casf2016_af3_test.csv"),
            "af3_subdir": "casf2016",
        },
        "0ligandbias": {
            "benchmark_csv": os.path.join(
                args.eval_dir, "0ligandbias", "0ligandbias_test.csv"),
            "af3_csv": os.path.join(
                args.eval_dir, "0ligandbias", "0ligandbias_af3_test.csv"),
            "af3_subdir": "0ligandbias",
        },
        "oodtest": {
            "benchmark_csv": os.path.join(
                args.eval_dir, "ood-test", "oodtest_test.csv"),
            "af3_csv": os.path.join(
                args.eval_dir, "ood-test", "oodtest_af3_test.csv"),
            "af3_subdir": "oodtest",
        },
    }

    all_results = []
    all_stats = {}

    for benchmark in args.benchmarks:
        if benchmark not in benchmark_config:
            logging.warning(f"Unknown benchmark: {benchmark}")
            continue

        config = benchmark_config[benchmark]
        bench_csv = config["benchmark_csv"]

        if not os.path.exists(bench_csv):
            # Try alternate locations
            alt_paths = [
                os.path.join(args.eval_dir, benchmark, f"{benchmark}_test.csv"),
                os.path.join(args.eval_dir, f"{benchmark}_test.csv"),
            ]
            found = False
            for alt in alt_paths:
                if os.path.exists(alt):
                    bench_csv = alt
                    found = True
                    break
            if not found:
                logging.error(f"Benchmark CSV not found: {bench_csv}")
                continue

        # Try finding AF3 test CSV
        af3_csv = config.get("af3_csv", "")
        if not af3_csv or not os.path.exists(af3_csv):
            # Look in the af3_dir
            af3_csv_alt = os.path.join(
                args.af3_dir, f"{config['af3_subdir']}_af3_test.csv"
            )
            if os.path.exists(af3_csv_alt):
                af3_csv = af3_csv_alt
            else:
                af3_csv = ""

        logging.info(f"\n{'='*60}")
        logging.info(f"Benchmark: {benchmark}")
        logging.info(f"CSV: {bench_csv}")
        logging.info(f"AF3 CSV: {af3_csv or 'not found'}")
        logging.info(f"{'='*60}")

        entries = load_benchmark_entries(
            bench_csv, af3_csv, args.data_root, args.af3_dir,
            config["af3_subdir"],
        )

        if args.limit > 0:
            entries = entries[:args.limit]

        logging.info(f"Entries to process: {len(entries)}")

        protein_rmsds = []
        ligand_rmsds = []
        benchmark_results = []

        for i, entry in enumerate(entries):
            pdb_id = entry["pdb_id"]
            if (i + 1) % 50 == 0 or args.verbose:
                logging.info(f"  [{i+1}/{len(entries)}] {pdb_id}")

            result = compute_entry_rmsd(
                af3_protein_path=entry["af3_protein"],
                af3_ligand_path=entry["af3_ligand_sdf"],
                xtal_protein_path=entry["xtal_protein"],
                xtal_ligand_path=entry["xtal_ligand_sdf"],
                pdb_id=pdb_id,
                ccd_code=entry.get("ccd_code"),
            )

            row = {
                "pdb_id": pdb_id,
                "benchmark": benchmark,
                "pk": entry.get("pk", ""),
                "protein_ca_rmsd": (f"{result['protein_rmsd']:.4f}"
                                    if result['protein_rmsd'] is not None
                                    else ""),
                "ligand_rmsd": (f"{result['ligand_rmsd']:.4f}"
                                if result['ligand_rmsd'] is not None
                                else ""),
                "n_ca_matched": result["n_ca_matched"],
                "n_ligand_atoms_matched": result["n_ligand_atoms_matched"],
                "n_af3_chains": result["n_af3_chains"],
                "n_xtal_chains": result["n_xtal_chains"],
                "chain_matches": result["chain_matches"],
                "warnings": "; ".join(result["warnings"]),
            }
            benchmark_results.append(row)
            all_results.append(row)

            if result["protein_rmsd"] is not None:
                protein_rmsds.append(result["protein_rmsd"])
            if result["ligand_rmsd"] is not None:
                ligand_rmsds.append(result["ligand_rmsd"])

        # ── Statistics ──
        prot_stats = compute_statistics(protein_rmsds, "protein_ca")
        lig_stats = compute_statistics(ligand_rmsds, "ligand")

        all_stats[benchmark] = {
            "protein_ca": prot_stats,
            "ligand": lig_stats,
            "total_entries": len(entries),
            "protein_computed": len(protein_rmsds),
            "ligand_computed": len(ligand_rmsds),
            "failed": len(entries) - len(protein_rmsds),
        }

        # Print summary
        print(f"\n{'─'*60}")
        print(f"  {benchmark} — Summary")
        print(f"{'─'*60}")
        print(f"  Total entries:          {len(entries)}")
        print(f"  Protein RMSD computed:  {len(protein_rmsds)}")
        print(f"  Ligand RMSD computed:   {len(ligand_rmsds)}")
        if prot_stats["n"] > 0:
            print(f"  Protein Cα RMSD:")
            print(f"    Mean:   {prot_stats['mean']:.3f} Å")
            print(f"    Median: {prot_stats['median']:.3f} Å")
            print(f"    Std:    {prot_stats['std']:.3f} Å")
            print(f"    Range:  [{prot_stats['min']:.3f}, {prot_stats['max']:.3f}] Å")
        if lig_stats["n"] > 0:
            print(f"  Ligand RMSD:")
            print(f"    Mean:   {lig_stats['mean']:.3f} Å")
            print(f"    Median: {lig_stats['median']:.3f} Å")
            print(f"    Std:    {lig_stats['std']:.3f} Å")
            print(f"    Range:  [{lig_stats['min']:.3f}, {lig_stats['max']:.3f}] Å")
            print(f"    < 2 Å:  {lig_stats['pct_below_2A']:.1f}%")
            print(f"    < 5 Å:  {lig_stats['pct_below_5A']:.1f}%")

    # ── Write results CSV ──
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = [
        "pdb_id", "benchmark", "pk",
        "protein_ca_rmsd", "ligand_rmsd",
        "n_ca_matched", "n_ligand_atoms_matched",
        "n_af3_chains", "n_xtal_chains",
        "chain_matches", "warnings",
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults CSV: {args.output_csv}")

    # ── Write statistics CSV ──
    stats_rows = []
    for benchmark, stats in all_stats.items():
        for metric in ["protein_ca", "ligand"]:
            s = stats[metric]
            stats_rows.append({
                "benchmark": benchmark,
                "metric": metric,
                "n": s["n"],
                "mean": f"{s['mean']:.4f}" if s["mean"] is not None else "",
                "median": f"{s['median']:.4f}" if s["median"] is not None else "",
                "std": f"{s['std']:.4f}" if s["std"] is not None else "",
                "min": f"{s['min']:.4f}" if s["min"] is not None else "",
                "max": f"{s['max']:.4f}" if s["max"] is not None else "",
                "pct_below_2A": (f"{s['pct_below_2A']:.1f}"
                                 if s.get("pct_below_2A") is not None else ""),
                "pct_below_5A": (f"{s['pct_below_5A']:.1f}"
                                 if s.get("pct_below_5A") is not None else ""),
                "total_entries": stats["total_entries"],
                "computed": stats[f"{metric.split('_')[0]}_computed"]
                            if metric == "protein_ca"
                            else stats["ligand_computed"],
                "failed": stats["failed"],
            })
    with open(args.output_stats, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "benchmark", "metric", "n", "mean", "median", "std",
            "min", "max", "pct_below_2A", "pct_below_5A",
            "total_entries", "computed", "failed",
        ])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"Statistics CSV: {args.output_stats}")

    # ── Final cross-benchmark summary ──
    print(f"\n{'═'*60}")
    print(f"  CROSS-BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    for benchmark, stats in all_stats.items():
        ps = stats["protein_ca"]
        ls = stats["ligand"]
        print(f"\n  {benchmark}:")
        if ps["n"] > 0:
            print(f"    Protein Cα RMSD: {ps['mean']:.2f} ± {ps['std']:.2f} Å "
                  f"(median {ps['median']:.2f}, n={ps['n']})")
        if ls["n"] > 0:
            print(f"    Ligand RMSD:     {ls['mean']:.2f} ± {ls['std']:.2f} Å "
                  f"(median {ls['median']:.2f}, n={ls['n']}, "
                  f"<2Å: {ls['pct_below_2A']:.0f}%, <5Å: {ls['pct_below_5A']:.0f}%)")


if __name__ == "__main__":
    main()