"""An independently implemented spatial effect, usable inside Compose."""

from dataclasses import dataclass

import numpy as np

from anamorph import Image, SpatialEffect
from anamorph.types import validate_image


@dataclass(frozen=True)
class ChannelPermutation(SpatialEffect):
    """An artistic color effect. This does not obscure facial structure."""

    def apply(self, image: Image) -> Image:
        validate_image(image)
        return np.ascontiguousarray(image[..., [1, 2, 0]])
