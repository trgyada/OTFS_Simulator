# =============================================================================
# modulation.py — Gray-coded 16-QAM modülasyon / demodülasyon
# =============================================================================

import numpy as np
from config import bits_per_symbol

QAM16_MAP = {
    (0, 0, 0, 0): -3 + 3j,
    (0, 0, 0, 1): -1 + 3j,
    (0, 0, 1, 1):  1 + 3j,
    (0, 0, 1, 0):  3 + 3j,

    (0, 1, 0, 0): -3 + 1j,
    (0, 1, 0, 1): -1 + 1j,
    (0, 1, 1, 1):  1 + 1j,
    (0, 1, 1, 0):  3 + 1j,

    (1, 1, 0, 0): -3 - 1j,
    (1, 1, 0, 1): -1 - 1j,
    (1, 1, 1, 1):  1 - 1j,
    (1, 1, 1, 0):  3 - 1j,

    (1, 0, 0, 0): -3 - 3j,
    (1, 0, 0, 1): -1 - 3j,
    (1, 0, 1, 1):  1 - 3j,
    (1, 0, 1, 0):  3 - 3j,
}

_NORM = np.sqrt(10)


def qam16_modulation(bits):
    if len(bits) % bits_per_symbol != 0:
        raise ValueError("Bit sayısı 4'ün katı olmalıdır.")

    bit_groups = bits.reshape(-1, 4)
    symbols = np.array(
        [QAM16_MAP[tuple(group)] / _NORM for group in bit_groups],
        dtype=complex,
    )
    return bit_groups, symbols


def qam16_demodulation(symbols):
    def i_bits(x):
        if x < -2 / _NORM:
            return [0, 0]
        elif x < 0:
            return [0, 1]
        elif x < 2 / _NORM:
            return [1, 1]
        else:
            return [1, 0]

    def q_bits(y):
        if y > 2 / _NORM:
            return [0, 0]
        elif y > 0:
            return [0, 1]
        elif y > -2 / _NORM:
            return [1, 1]
        else:
            return [1, 0]

    bits_out = []
    for s in symbols:
        bits_out.extend(q_bits(np.imag(s)) + i_bits(np.real(s)))
    return np.array(bits_out)