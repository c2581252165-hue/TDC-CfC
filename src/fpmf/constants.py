from __future__ import annotations

HISTORY_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
TARGET_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
HISTORY_COLUMNS = tuple(f"S2_{band}" for band in HISTORY_BANDS)
TARGET_COLUMNS = tuple(f"S2_{band}" for band in TARGET_BANDS)
TARGET_FROM_HISTORY = tuple(HISTORY_BANDS.index(band) for band in TARGET_BANDS)

LOOKBACK = 12
HORIZON = 1
EXPECTED_POINT_COUNT = 9596
EXPECTED_MONTH_COUNT = 65
OVERLAP_DIFFERENCE_LIMIT = 0.05
HISTORY_MIN_VALID_ACQUISITIONS = 1
TARGET_MIN_VALID_ACQUISITIONS = 2

SEEDS = (438344685, 293280205, 353421717)
MIN_POINT_MONTHS_FOR_R2 = 8

PROHIBITED_FEATURE_TOKENS = (
    "coord_point_id",
    "point_id",
    "longitude",
    "latitude",
    "grid_x",
    "grid_y",
    "easting",
    "northing",
    "dem",
    "elevation",
    "land_cover",
    "distance",
    "coast",
    "target_quality",
)

# Sentinel-2A nominal central wavelength and bandwidth values (nm). They are
# fixed sensor metadata, not learned site attributes and not a continuous-
# spectrum claim. K05 replaces these descriptors with ordinary band IDs.
S2_BAND_METADATA_NM = {
    "B2": (492.4, 66.0),
    "B3": (559.8, 36.0),
    "B4": (664.6, 31.0),
    "B5": (704.1, 15.0),
    "B6": (740.5, 15.0),
    "B7": (782.8, 20.0),
    "B8": (832.8, 106.0),
    "B8A": (864.7, 21.0),
    "B11": (1613.7, 91.0),
    "B12": (2202.4, 175.0),
}

SPECTRAL_GROUPS = {
    "VIS": (0, 1, 2),
    "RED_EDGE_NIR": (3, 4, 5, 6, 7),
    "SWIR": (8, 9),
}
