# =============================================================================
# config.py — OTFS simülasyonu için temel çerçeve parametreleri.
# Bu dosyadaki sabitler tüm modüller tarafından import edilerek kullanılır.
# =============================================================================

# Delay (gecikme) eksenindeki sembol sayısı.
M = 4

# Doppler eksenindeki sembol sayısı.
N = 4

# Her OFDM-benzeri blok için Cyclic Prefix (döngüsel önek) uzunluğu.
cp_len = 1

# 16-QAM modülasyonu: her sembol 4 bit taşır.
bits_per_symbol = 4

# Tek bir OTFS çerçevesi için toplam bit sayısı.
num_bits = M * N * bits_per_symbol
