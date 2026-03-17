# =============================================================================
# modulation.py — 16-QAM modülasyon ve demodülasyon fonksiyonları.
# Gray-kodlu konstelasyon haritası; ortalama sembol gücü sqrt(10) ile normalize edilir.
# =============================================================================

import numpy as np
from config import bits_per_symbol

# Gray-kodlu 16-QAM konstelasyon haritası: 4-bit tuple → karmaşık sembol.
# Normalize edilmemiş koordinatlar; gönderirken sqrt(10)'a bölünür.
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

# Normalizasyon sabiti: E[|s|²] = 1 olması için sqrt(10).
_NORM = np.sqrt(10)


def qam16_modulation(bits):
    """
    Bit dizisini 16-QAM sembollerine dönüştürür.

    Parametreler
    ------------
    bits : np.ndarray
        Uzunluğu 4'ün katı olan 0/1 bit dizisi.

    Döndürür
    --------
    bit_quads : np.ndarray, şekil (N, 4)
        4'erli gruplar halinde ayrılmış giriş bitleri.
    symbols : np.ndarray, karmaşık
        Normalize edilmiş 16-QAM sembolleri.
    """
    if len(bits) % bits_per_symbol != 0:
        raise ValueError("Bit sayısı 4'ün katı olmalıdır (16-QAM).")

    # Bitleri 4'erli gruplara ayır.
    bit_quads = bits.reshape(-1, 4)

    # Her grubu konstelasyon haritasından karşılık gelen sembole çevir.
    symbols = np.array(
        [QAM16_MAP[tuple(q)] / _NORM for q in bit_quads],
        dtype=complex,
    )

    return bit_quads, symbols


def qam16_demodulation(symbols):
    """
    16-QAM sembollerinden bit dizisi geri çıkarır (hard-decision).

    Parametreler
    ------------
    symbols : np.ndarray, karmaşık
        Normalize edilmiş (veya gürültülü) 16-QAM sembolleri.

    Döndürür
    --------
    bits : np.ndarray
        Demodüle edilmiş 0/1 bit dizisi.
    """

    def _i_bits(x):
        """In-phase (gerçek) eksen karar eşikleri → 2 bit."""
        if x < -2 / _NORM:
            return [0, 0]
        elif x < 0:
            return [0, 1]
        elif x < 2 / _NORM:
            return [1, 1]
        else:
            return [1, 0]

    def _q_bits(y):
        """Quadrature (sanal) eksen karar eşikleri → 2 bit (yukarıdan aşağıya)."""
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
        # Konstelasyon haritasındaki bit sırasıyla uyumlu: önce Q, sonra I bitleri.
        bits_out.extend(_q_bits(np.imag(s)) + _i_bits(np.real(s)))

    return np.array(bits_out)
