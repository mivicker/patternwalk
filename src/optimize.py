import numpy as np
from math import comb, floor, sqrt

def roll_over(num, upper_bound):
    return num % upper_bound


def bin_to_int(num):
    return int(num, 2)


def all_steps(upper_bound):
    i = 0
    while 2 ** i < upper_bound:
        yield bin(2 ** i)
        i += 1


def num_partitions(n, k):
    return int(comb(n + k - 1, k - 1))


def all_possibles(item, upper_bound):
    return {
        bin(roll_over(item ^ bin_to_int(step), bin_to_int("1100")))
        for step in all_steps(bin_to_int("1100"))
    }


def number_to_base(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


if __name__ == "__main__":
    # three products, need to chose 2 total, so four options
    """
    x [0,0,0]
    x [0,0,1]
    [0,0,2]
    x [0,1,0]
    [0,1,1]
    x [0,1,2]
    [0,2,0]
    x [0,2,1]
    x [0,2,2]
    x [1,0,0]
    [1,0,1]
    x [1,0,2]
    [1,1,0]
    x [1,1,1]
    x [1,2,0]
    x [1,2,1]
    x [1,2,2]
    [2,0,0]
    """
   
    # sums of two powers of three
    # how can we make this for any n and k
    n = 4 # number products
    k = 3 # number to choose
    q = 0
    pows = (k+1)**np.arange(8)
    combos = pows + pows.reshape(-1,1)
    to_check = combos[np.triu_indices(pows.shape[0])] + (k - 2)
    result = []
    for i in range(400):
        if sum(number_to_base(i, k+1)) == k:
            result.append(i)
            print(q, end = '\t')
            if i in to_check:
                print(i, end="\t")
            print(i)
            q += 1
# for i in range(bin_to_int("1111")):
#        print(all_possibles(i, bin_to_int("1010")))

# when k == 2, matches sums of powers of 3.
