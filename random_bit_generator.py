import random
def generate_random_bits(num_bits):
    return [random.randint(0, 1) for _ in range(num_bits)]