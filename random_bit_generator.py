import random


def generate_random_bits(num_bits):
    # Her eleman 0 veya 1 olan Python listesi dondur.
    return [random.randint(0, 1) for _ in range(num_bits)]
