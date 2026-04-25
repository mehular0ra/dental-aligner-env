"""
Clinical profile loader for real patient data from Tsinghua dataset.
Loads 1,063 patient profiles and provides sampling by diagnosis/difficulty.
"""
import json
import os
from typing import Dict, List, Optional, Any

# Path to the Tsinghua case database
_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets", "tsinghua", "case_database.json",
)

_PROFILES: Optional[Dict[str, dict]] = None


def _load_profiles() -> Dict[str, dict]:
    global _PROFILES
    if _PROFILES is None:
        with open(_DB_PATH, "r") as f:
            _PROFILES = json.load(f)
    return _PROFILES


def get_all_profiles() -> Dict[str, dict]:
    return _load_profiles()


def sample_profile(
    difficulty: str,
    rng,  # np.random.Generator
    malocclusion: Optional[str] = None,
) -> dict:
    """
    Sample a patient profile matching the requested difficulty.
    Optionally filter by malocclusion class (ClassI, ClassII, ClassIII).
    """
    profiles = _load_profiles()
    candidates = []

    for pid, prof in profiles.items():
        if prof.get("difficulty_level") == difficulty:
            if malocclusion and prof.get("malocclusion") != malocclusion:
                continue
            candidates.append(prof)

    # If no exact difficulty match, use all profiles
    if not candidates:
        candidates = list(profiles.values())

    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


# Geometry perturbation parameters keyed by clinical diagnosis
MALOCCLUSION_GEOMETRY = {
    "ClassI": {
        # Class I: normal molar relationship, crowding/spacing issues
        "molar_shift_mm": 0.0,
        "incisor_overlap_mm": 0.0,
        "arch_compression": 0.0,
    },
    "ClassII": {
        # Class II: upper molars anterior to lower (distocclusion)
        "molar_shift_mm": 3.0,     # upper first molars 3mm anterior
        "incisor_overlap_mm": 2.0,  # increased overjet
        "arch_compression": 0.0,
    },
    "ClassIII": {
        # Class III: lower molars anterior to upper (mesiocclusion)
        "molar_shift_mm": -2.5,    # lower first molars 2.5mm anterior
        "incisor_overlap_mm": -1.5,  # negative overjet (underbite)
        "arch_compression": 0.0,
    },
}

CROWDING_PARAMS = {
    "Crowding_above_4": {"arch_compression": 0.10},  # compress arch 10%
    "Crowding_below_4": {"arch_compression": 0.05},   # compress arch 5%
}

OVERBITE_PARAMS = {
    "Deep_overbite": {"vertical_overlap_mm": 4.0},
    "Normal_overbite": {"vertical_overlap_mm": 1.5},
}

OVERJET_PARAMS = {
    "Deep_overjet": {"horizontal_protrusion_mm": 3.0},
    "Normal_overjet": {"horizontal_protrusion_mm": 1.0},
}
