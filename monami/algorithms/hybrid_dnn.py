"""Hybrid Position DNN: absolute (X, Y) prepended to relative neighbor features."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from monami.algorithms.monami_dnn import MonamiDNNAlgorithm
from monami.config import MLConfig
from monami.features import hybrid_feature_dim
from monami.ml import split_samples


class HybridPositionDNNAlgorithm(MonamiDNNAlgorithm):
    id = "4_Hybrid_Position"
    name = "Hybrid Position"
    description = (
        "Hybrid DNN: normalized absolute X, Y plus relative neighbor features "
        "(dX, dY, D, V × n)."
    )
    include_target_xy = True
    long_description = """
### Hybrid Position — absolute + relative neighbor DNN

Combines **Absolute Position** coordinates with **Relative Position** neighbor
features so the network can learn which cues matter most.

**Training features**
- Normalized absolute **X** and **Y** of the target (same scaling as Absolute Position)
- Then, for each of the **n** nearest training samples: **dX**, **dY**, **D**, **V**
- Total input dimension = `2 + 4 × n` (configure **n** below)

**Important details**
- Absolute location and relative neighborhood are both available to the model
- The neighbor pool is the **training split only** (test points are for validation labels)
- During sequential simulation, each newly drawn cell is added to the conditioning
  pool so later path cells see an evolving neighborhood (same as Relative Position)

**Prediction / simulation**
- Most-likely map = argmax of the softmax given hybrid features
- Sequential simulation samples from the softmax and grows the conditioning set
""".strip()

    def render_config_ui(
        self,
        st_module: Any,
        samples_df: pd.DataFrame,
        categorized_2d,
        *,
        random_seed: int,
        default_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        default_config = default_config or {}
        train_pool, _ = split_samples(samples_df, 0.2, seed=random_seed)
        max_neighbors = max(1, len(train_pool) - 1)
        default_n = min(int(default_config.get("n_nearest", MLConfig().n_nearest)), max_neighbors)

        n_nearest = st_module.number_input(
            "Nearest neighbors (n)",
            min_value=1,
            max_value=max_neighbors,
            value=default_n,
            help=(
                "Number of closest **training** samples used for relative features "
                "(dX, dY, D, V per neighbor), in addition to absolute X, Y. "
                f"Maximum is one less than the training pool size (currently **{max_neighbors}**)."
            ),
            key="hybrid_n_nearest",
        )
        st_module.caption(
            f"Input dimension = {hybrid_feature_dim(int(n_nearest))} "
            f"(X, Y + 4×{int(n_nearest)} neighbor features)."
        )
        return {"n_nearest": int(n_nearest)}
