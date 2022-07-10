from itertools import combinations
import numpy as np
from scipy.spatial.distance import cosine as cos_sim


def partitions(n, k):
    """
    Returns all possible arrangements of n indistinct objects into
    k distinct bins.
    """
    return np.array(
        [
            [b - a - 1 for a, b in zip((-1,) + c, c + (n + k - 1,))]
            for c in combinations(range(n + k - 1), k - 1)
        ]
    )


def walk_towards(
    target_vector: np.array, n_steps: int, possible_moves: np.array
) -> np.array:
    """
    Walks as far toward the target vector as possible using only vectors
    given in possible_moves array within n_steps.
    """
    position = np.zeros(target_vector.shape[0])
    for _ in range(n_steps):
        heading = target_vector - position
        sims = np.apply_along_axis(
            lambda array: cos_sim(heading, array), 1, possible_moves
        )
        step = possible_moves[np.argmin(sims), :]
        position = position + step

    return position
