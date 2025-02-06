from collections.abc import Sequence

import numpy as np
import pingouin as pg


def bootstrap_mean(data: Sequence[float]) -> dict[str, float]:
    ci, dist = pg.compute_bootci(
        data,
        func="mean",
        method="norm",
        n_boot=1000,
        decimals=3,
        seed=42,
        return_dist=True,
    )
    mean = np.mean(data)
    stderr = (ci[1] - ci[0]) / 2  # force symmetrization to get rid of tiny differences

    return {"mean": mean, "stderr": stderr}
