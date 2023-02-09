from typing import Iterable
from matplotlib.artist import Artist
import numpy as np


class Effect:
    """All effects should be subclassed from this base class, which provides
    a template that enforces the contract between effects and performance.

    The only requirement placed on an effect is that the get_image function consumes specific inputs
    (which are provided by a Performance object), and outputs an Iterable of Artist objects
    """

    def get_image(self, ax, x_data: np.array, y_data: np.array) -> Iterable[Artist]:
        raise NotImplementedError()
