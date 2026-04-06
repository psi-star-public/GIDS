"""
GIDS Observer Framework

This package turns the operational math from Parts 2-4 of the paper into
small, inspectable Python components plus toy experiments.
"""

from .references import PAPER_REFERENCE_TABLE
from .math_utils import sigmoid, softmax, binary_cross_entropy_from_logits
from .embedding import salience_slice, estimate_slow_embedding
from .categorical import (
    contextual_lift,
    build_event_categorical_embedding,
    build_slow_bank,
    build_fast_pool,
)
from .state import (
    OperationalState,
    world_model_step,
    readout_probability,
    best_proposition,
    ema_slow_update,
)
from .objective import total_loss

__all__ = [
    "PAPER_REFERENCE_TABLE",
    "sigmoid",
    "softmax",
    "binary_cross_entropy_from_logits",
    "salience_slice",
    "estimate_slow_embedding",
    "contextual_lift",
    "build_event_categorical_embedding",
    "build_slow_bank",
    "build_fast_pool",
    "OperationalState",
    "world_model_step",
    "readout_probability",
    "best_proposition",
    "ema_slow_update",
    "total_loss",
]
