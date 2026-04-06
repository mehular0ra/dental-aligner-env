"""
Dental constants for the aligner trajectory planner environment.
FDI tooth numbering system (28 teeth, no wisdom teeth).
"""

# FDI tooth numbering system (28 teeth, no wisdom teeth)
TOOTH_IDS = [
    11, 12, 13, 14, 15, 16, 17,  # upper right
    21, 22, 23, 24, 25, 26, 27,  # upper left
    31, 32, 33, 34, 35, 36, 37,  # lower left
    41, 42, 43, 44, 45, 46, 47,  # lower right
]

N_TEETH = 28
N_STAGES = 24

# Tooth type classification
TOOTH_TYPES = {
    11: 'central_incisor', 12: 'lateral_incisor', 13: 'canine',
    14: 'premolar_1',      15: 'premolar_2',       16: 'molar_1',
    17: 'molar_2',
    21: 'central_incisor', 22: 'lateral_incisor', 23: 'canine',
    24: 'premolar_1',      25: 'premolar_2',       26: 'molar_1',
    27: 'molar_2',
    31: 'central_incisor', 32: 'lateral_incisor', 33: 'canine',
    34: 'premolar_1',      35: 'premolar_2',       36: 'molar_1',
    37: 'molar_2',
    41: 'central_incisor', 42: 'lateral_incisor', 43: 'canine',
    44: 'premolar_1',      45: 'premolar_2',       46: 'molar_1',
    47: 'molar_2',
}

# Clinical per-stage movement limits (per tooth)
MAX_TRANSLATION_PER_STAGE_MM = 0.25   # max 0.25 mm translation per tooth per stage
MAX_ROTATION_PER_STAGE_DEG  = 2.0    # max 2.0 degrees rotation per tooth per stage

# Max total treatment distances (mm / degrees) per tooth type
TREATMENT_LIMITS = {
    'central_incisor': {'max_trans': 5.0, 'max_rot': 25.0},
    'lateral_incisor': {'max_trans': 4.5, 'max_rot': 25.0},
    'canine':          {'max_trans': 4.0, 'max_rot': 20.0},
    'premolar_1':      {'max_trans': 3.5, 'max_rot': 15.0},
    'premolar_2':      {'max_trans': 3.5, 'max_rot': 15.0},
    'molar_1':         {'max_trans': 2.5, 'max_rot': 10.0},
    'molar_2':         {'max_trans': 2.0, 'max_rot':  8.0},
}

# Build arch adjacency programmatically: teeth that are neighbours in same arch
def _build_adjacency():
    pairs = []
    # upper right: 11-17
    upper_right = [11, 12, 13, 14, 15, 16, 17]
    # upper left: 21-27
    upper_left = [21, 22, 23, 24, 25, 26, 27]
    # lower left: 31-37
    lower_left = [31, 32, 33, 34, 35, 36, 37]
    # lower right: 41-47
    lower_right = [41, 42, 43, 44, 45, 46, 47]

    for arch in [upper_right, upper_left, lower_left, lower_right]:
        for i in range(len(arch) - 1):
            pairs.append((arch[i], arch[i + 1]))

    # Cross-arch midline connections: 11-21, 31-41
    pairs.append((11, 21))
    pairs.append((31, 41))
    return pairs

ARCH_ADJACENCY = _build_adjacency()

# Opposing tooth pairs (upper-lower contact)
OPPOSING_PAIRS = {
    11: 41, 12: 42, 13: 43, 14: 44, 15: 45, 16: 46, 17: 47,
    21: 31, 22: 32, 23: 33, 24: 34, 25: 35, 26: 36, 27: 37,
}

# Staging priority order (teeth that should move first in treatment)
STAGING_PRIORITY = [
    'central_incisor',  # crowd relief first
    'lateral_incisor',
    'canine',
    'premolar_1',
    'premolar_2',
    'molar_1',          # molars anchor last
    'molar_2',
]

# Ideal upper arch positions (parabolic, 8mm spacing)
IDEAL_UPPER_TX = [35, 28, 20, 12, 4, -4, -12,    # quadrant 1 (right)
                  -35, -28, -20, -12, -4, 4, 12]  # quadrant 2 (left)
IDEAL_UPPER_TY = [0, 4, 9, 13, 16, 18, 19,
                  0, 4, 9, 13, 16, 18, 19]
IDEAL_UPPER_TZ = [0.0] * 14

IDEAL_LOWER_TX = [35, 28, 20, 12, 4, -4, -12,
                  -35, -28, -20, -12, -4, 4, 12]
IDEAL_LOWER_TY = [0, 4, 9, 13, 16, 18, 19,
                  0, 4, 9, 13, 16, 18, 19]
IDEAL_LOWER_TZ = [-2.0] * 14
