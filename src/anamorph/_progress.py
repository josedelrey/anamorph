"""Shared frame progress display for the sequential video passes."""

from __future__ import annotations

import math

from tqdm import tqdm


def frame_progress(description: str, *, total: int | float | None, enabled: bool = True) -> tqdm:
    """Use a counter when the decoder cannot provide a usable frame estimate."""
    count = int(total) if total is not None and math.isfinite(total) and total > 0 else None
    return tqdm(
        total=count, desc=description, unit="frame", dynamic_ncols=True, disable=not enabled
    )
