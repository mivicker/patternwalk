from itertools import combinations
from collections import deque
import numpy as np
from scipy.spatial.distance import cosine as cos_sim
from scipy.special import comb

MAX_ITERS = 1000


def calc_best_heading(heading, possible_moves):
    sims = np.apply_along_axis(lambda array: cos_sim(heading, array), 1, possible_moves)
    return sims.min(), possible_moves[np.argmin(sims), :]


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


def slow_partitions(n, k):
    """
    Returns all possible arrangements of n indistinct objects into
    k distinct bins.
    """
    for c in combinations(range(n + k - 1), k - 1):
        yield [b - a - 1 for a, b in zip((-1,) + c, c + (n + k - 1,))]


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
        step = calc_best_heading(heading, possible_moves)
        position = position + step

    return position


def find_adjacent_positions(vector, num_items):
    num_to_include = len(vector)
    steps_up = (vector < (num_items - 1) * np.eye(num_to_include)).astype(int)
    steps_down = -((vector > 0) * np.eye(num_to_include)).astype(int)
    adjacent = np.concatenate([steps_up, steps_down])
    return adjacent[adjacent.sum(axis=1) != 0]


def flip_out(vector, num_items):
    result = np.zeros(num_items)
    unique, counts = np.unique(vector, return_counts=True)
    result[unique] = counts
    return result


def flip_in(vector):
    result = []
    for i, count in enumerate(vector):
        result.extend([i] * int(count))
    return result


num_items = 16
num_to_include = 8

vector = np.random.randint(num_items, size=num_to_include)

# Available steps
# If vec_n == 0 add one only, if vec_n == 15 subtract one only
# else add one or subtract one.

target = np.array([13] * 30 + [12] * 10)

result = vector + find_adjacent_positions(vector, num_items)
possible_steps = np.apply_along_axis(
    lambda array: flip_out(array, num_items), 1, result
)


def walk_to_optimum(target, num_items, current=None):
    """
    Need to not walk backwards, and avoid loops.
    """
    best_score = np.inf
    if current is None:
        vector = np.random.randint(num_items, size=num_to_include)
        current = flip_out(vector, num_items)
        last = current

    for _ in range(MAX_ITERS):
        possible_steps = np.array(flip_in(current)) + find_adjacent_positions(
            np.array(flip_in(current)), num_items
        )

        current_best_score, best_direction = calc_best_heading(
            flip_out(target, num_items),
            np.apply_along_axis(
                lambda array: flip_out(array, num_items), 1, possible_steps
            ),
        )

        if best_score < current_best_score:
            print("best found")
            break

        if np.array_equal(best_direction, last):
            print(possible_steps)
            print("cycle starting")
            print(f"moving from {current}")
            print(f"moving to {best_direction}")
            print(f"score equals {current_best_score}")
            print(f"old score was {best_score}")
            break

        best_score = current_best_score
        last = current
        current = best_direction

    return current


def num_partitions(n, k):
    return int(comb(n + k - 1, k - 1))

partitions = slow_partitions(3, 4)
print(list(combinations(range(5), 3)))

