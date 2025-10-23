from itertools import combinations


def first_minimal_hitting_set(sets):
    """
    Find the first minimal hitting set for a family of sets.
    `sets` is an iterable of sets (each containing hashable elements).
    Returns a set which hits all input sets, minimal in size.
    """

    universe = set().union(*sets)  # all elements appearing
    sets = list(sets)

    # Try hitting sets of increasing size
    for size in range(1, len(universe) + 1):
        for combo in combinations(universe, size):
            candidate = set(combo)
            # Check if candidate hits all sets
            if all(candidate & s for s in sets):
                return candidate
    return None  # no hitting set found (should not happen unless input is empty)

def all_minimal_hitting_sets(sets):
    universe = set().union(*sets)  # all elements appearing
    sets = list(sets)
    minimal_hitting_sets = []

    # Try hitting sets of increasing size
    for size in range(1, len(universe) + 1):
        if not minimal_hitting_sets:
            for combo in combinations(universe, size):
                candidate = set(combo)
                # Check if candidate hits all sets
                if all(candidate & s for s in sets):
                    minimal_hitting_sets.append(candidate)
        else:
            break
    return minimal_hitting_sets