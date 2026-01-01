from __future__ import annotations

from typing import TypeAlias

import numpy as np

AlphaMinus: TypeAlias = float
AlphaPlus: TypeAlias = float


def alpha_minus_plus_from_ratio_and_magnitude(
    ratio: float, magnitude: float
) -> tuple[AlphaMinus, AlphaPlus]:
    alpha_minus = magnitude / np.sqrt(ratio) + 1
    alpha_plus = magnitude * np.sqrt(ratio) + 1
    return alpha_minus, alpha_plus
