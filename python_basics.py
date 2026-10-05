from IPython.core.pylabtools import retina_figure
from unicodedata import decomposition

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput

def count_vowels(data: TextInput) -> int:
    text = data.value
    text = data.value.lower()
    vowels = "aeiou"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    char_counts = {}
    for char in text:
        if char in char_counts:
            return False
        char_counts[char] = True
    return True


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while(number >= 10):
        prod = 1
        for char in str(number):
            prod *= int(char)
        number = prod
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = {}
    d = 2
    while d * d <= number:
        if number % d == 0:
            factors[d] = factors.get(d, 0) + 1
            number //= d
        else:
            d += 1
    if number > 1:
        factors[number] = factors.get(number, 0) + 1
    return [f"({base}^{power})" for base, power in factors.items()]


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 1
    while cube_count > 0:
        cube_count -= k**2
        if cube_count == 0:
            return k
        k += 1
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = str(data.value)
    length = len(number)
    half_len = (length - 1) // 2
    left_side = number[:half_len]
    right_side = number[length - half_len:]
    left_sum = sum(int(char) for char in left_side)
    right_sum = sum(int(char) for char in right_side)
    return left_sum == right_sum
