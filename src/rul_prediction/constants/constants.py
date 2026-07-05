"""
Project-wide constants for the RUL prediction project.

WHY: Constants are values that should NEVER change at runtime (e.g. the
fixed column names in the NASA CMAPSS dataset, the number of sensors).
Separating these from `config.yaml` (which holds values users MIGHT want
to change, like batch size or file paths) is a deliberate architecture
decision: it prevents someone from "configuring" something that would
break the code if changed (e.g. renaming a sensor column).

HOW: Plain Python module-level constants, imported directly.

WHERE: Used in preprocessing, feature engineering, and EDA modules, e.g.
    from rul_prediction.constants.constants import SENSOR_COLUMNS
"""

# NASA CMAPSS dataset column structure
# Columns: unit_number, time_in_cycles, 3 operating settings, 21 sensors
INDEX_COLUMNS = ["unit_number", "time_in_cycles"]
OPERATIONAL_SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

ALL_COLUMNS = INDEX_COLUMNS + OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS

# CMAPSS sub-dataset identifiers (FD001-FD004 differ by number of
# operating conditions and fault modes)
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

RANDOM_SEED = 42

# Sensors empirically identified as near-constant in FD001 (std ≈ 0, range < 0.01).
# Verified by EDA (Phase 2): these carry no signal and must be dropped before modelling.
NON_INFORMATIVE_SENSORS = [
    "sensor_1",   # T2  — fan inlet temp, completely flat
    "sensor_5",   # P2  — fan inlet pressure, completely flat
    "sensor_6",   # P15 — bypass-duct pressure, completely flat
    "sensor_10",  # epr — engine pressure ratio, completely flat
    "sensor_16",  # farB — burner fuel-air ratio, completely flat
    "sensor_18",  # Nf_dmd — demanded fan speed, completely flat
    "sensor_19",  # PCNfR_dmd — demanded corrected fan speed, completely flat
]
