# Main colors
PINK = "#DC267F"
BLUE = "#785EF0"
YELLOW = "#FFB000"

# Grey shades
CHARCOAL = "#232023"
IRON = "#322D31"
GRAPHITE = "#594D5B"
GRAY = "#808080"
COIN = "#9897A9"

# Auxiliary colors
MALIBU = "#648FFF"
ORANGE = "#FE6100"

METHOD_COLORS = {
    "VespaG": PINK,
    "GEMME": BLUE,
    "VESPA": YELLOW,
    "TranceptEVE-L": GRAPHITE,
    "ESM-2 (3B)": GRAY,
    "SaProt (650M)": COIN,
    "AlphaMissense": CHARCOAL,
    "PoET": IRON
}

MULTILINE_LABELS = {
    "VespaG": "VespaG",
    "GEMME": "GEMME",
    "VESPA": "VESPA",
    "PoET": "PoET",
    "AlphaMissense": "Alpha\nMissense",
    "TranceptEVE-L": "Trancept\nEVE-L",
    "SaProt (650M)": "SaProt\n(650M)",
    "ESM-2 (3B)": "ESM-2\n(3B)",
}

MILLIMETER = 1 / 2.54 / 10
WIDTH = 180 * MILLIMETER
HEIGHT = 100 * MILLIMETER

BARLABEL_FONTSIZE = 8
XTICK_FONTSIZE = 7
PANEL_LABEL_FONTSIZE = 16

BARPLOT_KEYWORDS = {
    "errorbar": ("se", 1.96),
    "n_boot": 1000,
    "err_kws": {"linewidth": 1},
    "capsize": 0.2,
}