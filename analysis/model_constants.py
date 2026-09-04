#!/usr/bin/env python3
"""The emission model's constants, recorded from the source that uses them.

    python model_constants.py /path/to/results_dir
"""

from __future__ import annotations

import sys
from pathlib import Path

from facts import emit
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from state_exposure import (ATTRIBUTION_SPLIT, AUX_FRACTION_AT_SEA,
                            CO2_PER_FUEL_T, DEFAULT_SERVICE_SPEED_KN,
                            MAX_LOAD, MIN_LOAD, SFOC_G_PER_KWH)


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <results_dir>")
    emit(Path(sys.argv[1]), "constants", {
        "sfoc_g_per_kwh": SFOC_G_PER_KWH,
        "co2_per_fuel_t": CO2_PER_FUEL_T,
        "aux_fraction_pct": round(100 * AUX_FRACTION_AT_SEA),
        "load_min_pct": round(100 * MIN_LOAD),
        "load_max_pct": round(100 * MAX_LOAD),
        "attribution_split_pct": round(100 * ATTRIBUTION_SPLIT),
        "default_service_kn": DEFAULT_SERVICE_SPEED_KN,
    })


if __name__ == "__main__":
    main()
