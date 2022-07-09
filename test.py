import numpy as np
from walk import partitions, walk_towards


def test_partitions():
    array = partitions(2, 2)
    assert np.array_equal(array, np.array([[0, 2], [1, 1], [2, 0]]))


def test_walk_towards_obvious():
    target = np.array([100, 100, 100])
    possible_steps = np.array([[1, 1, 1], [50, 0, 0], [-40, -40, -50]])
    assert np.array_equal(walk_towards(target, 100, possible_steps), target)


def test_walk_towards_limited():
    target = np.array([100, 100, 100])
    possible_steps = np.array([[1, 1, 1], [2, 2, 0], [100, 100, -1]])
    assert np.array_equal(
        walk_towards(target, 10, possible_steps), np.array([10, 10, 10])
    )

# What happens when you nearly hit your target before the steps are used up?

if __name__ == "__main__":
    test_partitions()
    test_walk_towards_obvious()
    test_walk_towards_limited()
