"""Cell type name ↔ index mapping for RiboNN's 78 human targets.

Target columns are ordered as they appear in RiboNN's training data,
filtered by the pattern '^TE_'. Index 0 = first TE_ column, etc.
"""

from __future__ import annotations

# Ordered list matching RiboNN's 78-target human model output columns.
# Source: external/RiboNN/data/example_training_data.txt column order.
HUMAN_CELL_TYPES: list[str] = [
    "108T",                                        # 0
    "12T",                                         # 1
    "A2780",                                       # 2
    "A549",                                        # 3
    "BJ",                                          # 4
    "BRx-142",                                     # 5
    "C643",                                        # 6
    "CRL-1634",                                    # 7
    "Calu-3",                                      # 8
    "Cybrid_Cells",                                # 9
    "H1-hESC",                                     # 10
    "H1933",                                       # 11
    "H9-hESC",                                     # 12
    "HAP-1",                                       # 13
    "HCC_tumor",                                   # 14
    "HCC_adjancent_normal",                        # 15
    "HCT116",                                      # 16
    "HEK293",                                      # 17
    "HEK293T",                                     # 18
    "HMECs",                                       # 19
    "HSB2",                                        # 20
    "HSPCs",                                       # 21
    "HeLa",                                        # 22
    "HeLa_S3",                                     # 23
    "HepG2",                                       # 24
    "Huh-7.5",                                     # 25
    "Huh7",                                        # 26
    "K562",                                        # 27
    "Kidney_normal_tissue",                        # 28
    "LCL",                                         # 29
    "LuCaP-PDX",                                   # 30
    "MCF10A",                                      # 31
    "MCF10A-ER-Src",                               # 32
    "MCF7",                                        # 33
    "MD55A3",                                      # 34
    "MDA-MB-231",                                  # 35
    "MM1.S",                                       # 36
    "MOLM-13",                                     # 37
    "Molt-3",                                      # 38
    "Mutu",                                        # 39
    "OSCC",                                        # 40
    "PANC1",                                       # 41
    "PATU-8902",                                   # 42
    "PC3",                                         # 43
    "PC9",                                         # 44
    "Primary_CD4+_T-cells",                        # 45
    "Primary_human_bronchial_epithelial_cells",    # 46
    "RD-CCL-136",                                  # 47
    "RPE-1",                                       # 48
    "SH-SY5Y",                                     # 49
    "SUM159PT",                                    # 50
    "SW480TetOnAPC",                               # 51
    "T47D",                                        # 52
    "THP-1",                                       # 53
    "U-251",                                       # 54
    "U-343",                                       # 55
    "U2392",                                       # 56
    "U2OS",                                        # 57
    "Vero_6",                                      # 58
    "WI38",                                        # 59
    "WM902B",                                      # 60
    "WTC-11",                                      # 61
    "ZR75-1",                                      # 62
    "cardiac_fibroblasts",                         # 63
    "ccRCC",                                       # 64
    "early_neurons",                               # 65
    "fibroblast",                                  # 66
    "hESC",                                        # 67
    "human_brain_tumor",                           # 68
    "iPSC-differentiated_dopamine_neurons",        # 69
    "megakaryocytes",                              # 70
    "muscle_tissue",                               # 71
    "neuronal_precursor_cells",                    # 72
    "neurons",                                     # 73
    "normal_brain_tissue",                         # 74
    "normal_prostate",                             # 75
    "primary_macrophages",                         # 76
    "skeletal_muscle",                             # 77
]

# Build lookup dicts
_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(HUMAN_CELL_TYPES)}
_NAME_TO_INDEX_LOWER: dict[str, int] = {name.lower(): i for i, name in enumerate(HUMAN_CELL_TYPES)}


def cell_type_to_index(name: str) -> int:
    """Convert a cell type name to its RiboNN target index.

    Case-insensitive lookup. Raises ValueError if not found.
    """
    # Try exact match first
    if name in _NAME_TO_INDEX:
        return _NAME_TO_INDEX[name]
    # Try case-insensitive
    lower = name.lower()
    if lower in _NAME_TO_INDEX_LOWER:
        return _NAME_TO_INDEX_LOWER[lower]
    raise ValueError(
        f"Unknown cell type: '{name}'. "
        f"Available types: {', '.join(HUMAN_CELL_TYPES[:10])}... "
        f"(use --list-cell-types to see all)"
    )


def index_to_cell_type(index: int) -> str:
    """Convert a RiboNN target index to its cell type name."""
    if 0 <= index < len(HUMAN_CELL_TYPES):
        return HUMAN_CELL_TYPES[index]
    raise ValueError(f"Index {index} out of range (0-{len(HUMAN_CELL_TYPES)-1})")
