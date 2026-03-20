from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "action_best.h5"

ACTIONS = np.array([
    "cold",
    "fever",
    "cough",
    "medication",
    "injection",
    "operation",
    "pain",
])

SEQUENCE_LENGTH = 30
PREDICTION_STABILITY_WINDOW = 10
DEFAULT_THRESHOLD = 0.4
MAX_SENTENCE_LENGTH = 5

PROB_COLORS = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
]
