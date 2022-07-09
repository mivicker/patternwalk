# patternwalk
## functions

`partitions(n, k)`
Small modification from a function provided here https://stackoverflow.com/questions/28965734/general-bars-and-stars
Gives all possible arrangements of n indistinguishable objects into k distinguishable bins.
Returns an numpy array.

`walk_towards(target, n_steps, possible_moves)`
Returns an array of the nearest position to `target` reachable in `n_steps` using only vectors provided in `possible_moves`.

