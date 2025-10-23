import random
from typing import TypeVar, List, Callable
from spec_repair import config

T = TypeVar("T")
HeuristicType = Callable[[List[T]], T]


def choose_one_with_heuristic(options_list: List[T], heuristic_function: HeuristicType) -> T:
    assert len(options_list) > 0
    if not config.STATISTICS and len(options_list) == 1:  # No need for heuristic
        return options_list[0]
    return heuristic_function(options_list)


def manual_choice(options_list: List[T]) -> T:
    if not config.MANUAL:
        return random.choice(options_list)
    options_list.sort()
    print("Select an option by choosing its index:")
    for idx, option in enumerate(options_list):
        print(f"{idx}: {option}")

    while True:
        try:
            print(f"Enter the index of your choice [0-{len(options_list) - 1}]: ")
            choice = int(input())
            if 0 <= choice < len(options_list):
                return options_list[choice]
            else:
                print("Invalid index. Please enter a valid index.")
        except ValueError:
            print("Invalid input. Please enter a valid integer index.")


def random_choice(options_list: List[T]) -> T:
    return random.choice(options_list)


def first_choice(options_list: List[T]) -> T:
    return nth_choice(0, options_list)


def last_choice(options_list: List[T]) -> T:
    return nth_choice(-1, options_list)


def nth_choice(index: int, options_list: List[T]) -> T:
    # options_list.sort()
    assert len(options_list) >= index + 1
    return options_list[index]
