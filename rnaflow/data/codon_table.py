"""Standard genetic code: codon ↔ amino acid mapping and synonymous codon groups.

Used to constrain gradient inversion so that the CDS region only explores
synonymous codon substitutions (preserving the encoded protein).
"""

from __future__ import annotations

# Standard genetic code (RNA codons)
CODON_TO_AA: dict[str, str] = {
    "UUU": "F", "UUC": "F",
    "UUA": "L", "UUG": "L", "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I",
    "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S", "AGU": "S", "AGC": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y",
    "UAA": "*", "UAG": "*", "UGA": "*",
    "CAU": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C",
    "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Build reverse mapping: amino acid → list of synonymous codons
AA_TO_CODONS: dict[str, list[str]] = {}
for codon, aa in CODON_TO_AA.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

# Nucleotide to index (matching RiboNN: A=0, U/T=1, C=2, G=3)
NUC_IDX = {"A": 0, "U": 1, "C": 2, "G": 3}

# Human codon usage frequencies (per thousand) from Kazusa Codon Usage Database.
# Available for optional codon usage bias — not used by default.
HUMAN_CODON_FREQ: dict[str, float] = {
    "UUU": 17.6, "UUC": 20.3,
    "UUA":  7.7, "UUG": 12.9, "CUU": 13.2, "CUC": 19.6, "CUA":  7.2, "CUG": 39.6,
    "AUU": 16.0, "AUC": 20.8, "AUA":  7.5,
    "AUG": 22.0,
    "GUU": 11.0, "GUC": 14.5, "GUA":  7.1, "GUG": 28.1,
    "UCU": 15.2, "UCC": 17.7, "UCA": 12.2, "UCG":  4.4, "AGU": 12.1, "AGC": 19.5,
    "CCU": 17.5, "CCC": 19.8, "CCA": 16.9, "CCG":  6.9,
    "ACU": 13.1, "ACC": 18.9, "ACA": 15.1, "ACG":  6.1,
    "GCU": 18.4, "GCC": 27.7, "GCA": 15.8, "GCG":  7.4,
    "UAU": 12.2, "UAC": 15.3,
    "UAA":  1.0, "UAG":  0.8, "UGA":  1.6,
    "CAU": 10.9, "CAC": 15.1,
    "CAA": 12.3, "CAG": 34.2,
    "AAU": 17.0, "AAC": 19.1,
    "AAA": 24.4, "AAG": 31.9,
    "GAU": 21.8, "GAC": 25.1,
    "GAA": 29.0, "GAG": 39.6,
    "UGU": 10.6, "UGC": 12.6,
    "UGG": 13.2,
    "CGU":  4.5, "CGC": 10.4, "CGA":  6.2, "CGG": 11.4, "AGA": 12.2, "AGG": 12.0,
    "GGU": 10.8, "GGC": 22.2, "GGA": 16.5, "GGG": 16.5,
}

# Normalized per-amino-acid relative frequencies (sum to 1.0 within each synonymous group)
HUMAN_CODON_REL_FREQ: dict[str, dict[str, float]] = {}
for _aa, _codons in AA_TO_CODONS.items():
    _total = sum(HUMAN_CODON_FREQ.get(c, 1.0) for c in _codons)
    HUMAN_CODON_REL_FREQ[_aa] = {c: HUMAN_CODON_FREQ.get(c, 1.0) / _total for c in _codons}


def translate(cds_seq: str) -> str:
    """Translate an RNA CDS sequence to amino acids.

    Args:
        cds_seq: RNA sequence (must be multiple of 3 in length).

    Returns:
        Amino acid string.
    """
    cds_seq = cds_seq.upper().replace("T", "U")
    protein = []
    for i in range(0, len(cds_seq) - 2, 3):
        codon = cds_seq[i:i+3]
        aa = CODON_TO_AA.get(codon, "X")
        protein.append(aa)
    return "".join(protein)


def get_synonymous_codons(codon: str) -> list[str]:
    """Get all synonymous codons for the amino acid encoded by the given codon."""
    codon = codon.upper().replace("T", "U")
    aa = CODON_TO_AA.get(codon)
    if aa is None:
        return [codon]
    return AA_TO_CODONS[aa]
