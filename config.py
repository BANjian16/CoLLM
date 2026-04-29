PAPER_THRESHOLDS = {
    "FD001": {
        "A": (0.3, 0.1),
        "B": (0.4, 0.1),
        "C": (0.6, 0.05),
    },
    "FD003": {
        "A": (0.15, 0.1),
        "B": (0.4, 0.1),
        "C": (0.6, 0.05),
    },
}

DEFAULT_SUBSET = "FD001"
DEFAULT_PRESET = "C"
TAU1, TAU2 = PAPER_THRESHOLDS[DEFAULT_SUBSET][DEFAULT_PRESET]


def get_thresholds(subset=DEFAULT_SUBSET, preset=DEFAULT_PRESET):
    subset = subset.upper()
    preset = preset.upper()
    return PAPER_THRESHOLDS[subset][preset]
