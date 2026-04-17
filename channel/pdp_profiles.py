import numpy as np


def _profile(name, delays_us, powers_db, doppler_types):
    delays_us = np.asarray(delays_us, dtype = float)
    powers_db = np.asarray(powers_db, dtype = float)

    if len(delays_us) != len(powers_db):
        raise ValueError(f"{name}: delays_us and powers_db must have same length")
    if len(doppler_types) != len(delays_us):
        raise ValueError(f"{name}: doppler_types length mismatch")
    return {
        "name": name,
        "delays_s": delays_us * 1e-6,
        "powers_db": powers_db,
        "doppler_types": list(doppler_types),
    }
# Typical Urban
TU = _profile(
    "Typical Urban",
    delays_us=[0.0, 0.2, 0.5, 1.6, 2.3, 5.0],
    powers_db=[-3.0, 0.0, -2.0, -6.0, -8.0, -10.0],
    doppler_types=["CLASS", "CLASS", "CLASS", "GAUS1", "GAUS1", "GAUS2"],
)

# Rural Area
RA = _profile(
    "Rural Area",
    delays_us=[0.0, 0.2, 0.4, 0.6],
    powers_db=[0.0, -2.0, -10.0, -20.0],
    doppler_types=["RICE", "CLASS", "CLASS", "CLASS"],
)


# Hilly Terrain
HT = _profile(
    "Hilly Terrain",
    delays_us=[0.0, 0.2, 0.4, 0.6, 15.0, 17.2],
    powers_db=[0.0, -2.0, -4.0, -7.0, -6.0, -12.0],
    doppler_types=["CLASS", "CLASS", "CLASS", "CLASS", "GAUS2", "GAUS2"],
)

CHANNEL_PROFILES = {
    "RA": RA,
    "TU": TU,
    "HT": HT,
}