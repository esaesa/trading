# simulation.py
import numpy as np
from typing import Tuple

def apply_slippage(
    price: float,
    low: float,
    high: float,
    probability: float
) -> Tuple[float, float, float]:
    """
    SRP: Apply symmetric uniform slippage with the same semantics as before.
    - Trigger with probability `probability` (uses np.random.rand()).
    - Independently scale price, low, and high by U[0.995, 1.005].
    """
    if probability <= 0.0:
        return price, low, high

    if np.random.rand() < probability:
        return (
            price * np.random.uniform(0.995, 1.005),
            low   * np.random.uniform(0.995, 1.005),
            high  * np.random.uniform(0.995, 1.005),
        )
    return price, low, high
