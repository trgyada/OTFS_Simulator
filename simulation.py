# =============================================================================
# simulation.py — OTFS TX-RX pipeline ve Monte-Carlo SNR sweep fonksiyonları.
#
# Pipeline özeti:
#   TX: bits → 16-QAM → DD grid → ISFFT → Heisenberg → CP ekle → seri sinyal
#   RX: seri sinyal → CP at → Wigner → SFFT → DD grid → 16-QAM demod → bits
# =============================================================================

import numpy as np

import random_bit_generator
from config import M, N, cp_len, bits_per_symbol, num_bits
from modulation import qam16_modulation, qam16_demodulation
from channel import apply_channel


# ---------------------------------------------------------------------------
# Bit üretici
# ---------------------------------------------------------------------------

def generate_bits():
    """Tek bir OTFS çerçevesini dolduracak kadar rastgele bit üretir."""
    return np.array(random_bit_generator.generate_random_bits(num_bits))


# ---------------------------------------------------------------------------
# Tek çerçeve simülasyonu
# ---------------------------------------------------------------------------

def run_otfs_simulation(bits, channel_type="Ideal", snr_db=20.0):
    """
    Verilen bit dizisi üzerinde tam bir OTFS TX→Kanal→RX döngüsü çalıştırır.

    Parametreler
    ------------
    bits : np.ndarray
        M*N*bits_per_symbol uzunluğunda 0/1 bit dizisi.
    channel_type : str
        "Ideal" veya "AWGN".
    snr_db : float
        AWGN için SNR değeri (dB).

    Döndürür
    --------
    dict
        TX/RX aşamalarına ait tüm ara verileri ve BER/SER metriklerini içerir.
    """

    # ------------------------------------------------------------------
    # TX tarafı
    # ------------------------------------------------------------------

    # Bitleri 16-QAM sembollerine dönüştür.
    bit_groups, mod_symbols = qam16_modulation(bits)

    # Sembol sayısı çerçeve boyutuyla uyuşmalı.
    expected_symbols = M * N
    if len(mod_symbols) != expected_symbols:
        raise ValueError(
            f"{M}x{N} grid için {expected_symbols} sembol bekleniyor, "
            f"{len(mod_symbols)} geldi."
        )

    # Sembolleri Delay-Doppler (DD) gridine yerleştir: şekil (M, N).
    dd_grid = mod_symbols.reshape(M, N)

    # ISFFT benzeri dönüşüm: DD → Time-Frequency (TF) düzlemi.
    # Doppler ekseninde IFFT, gecikme ekseninde FFT uygulanır.
    X_tf = np.fft.fft(np.fft.ifft(dd_grid, axis=1), axis=0)

    # Heisenberg benzeri dönüşüm: TF → zaman blokları.
    x_time_blocks = np.fft.ifft(X_tf, axis=0)

    # Her zaman bloğuna Cyclic Prefix (CP) ekle.
    blocks_with_cp = []
    for n in range(N):
        block = x_time_blocks[:, n]
        cp = block[-cp_len:]                      # Bloğun son cp_len örneği.
        blocks_with_cp.append(np.concatenate([cp, block]))

    # Blokları sütun-öncelikli (Fortran) düzende seri hale getir.
    blocks_with_cp = np.array(blocks_with_cp).T
    tx_signal = blocks_with_cp.flatten(order="F")

    # ------------------------------------------------------------------
    # Kanal
    # ------------------------------------------------------------------

    rx_signal = apply_channel(tx_signal, channel_type=channel_type, snr_db=snr_db)

    # ------------------------------------------------------------------
    # RX tarafı
    # ------------------------------------------------------------------

    # Seri sinyali bloklara geri dönüştür; CP'yi at.
    rx_blocks_with_cp = rx_signal.reshape(M + cp_len, N, order="F")
    rx_blocks = rx_blocks_with_cp[cp_len:, :]

    # Wigner benzeri dönüşüm: zaman → TF düzlemi.
    Y_tf = np.fft.fft(rx_blocks, axis=0)

    # SFFT benzeri dönüşüm: TF → DD düzlemi.
    Y_dd = np.fft.fft(np.fft.ifft(Y_tf, axis=0), axis=1)

    # DD sembollerini bitlere demodüle et.
    rx_symbols = Y_dd.reshape(-1)
    rx_bits = qam16_demodulation(rx_symbols)

    # ------------------------------------------------------------------
    # Performans metrikleri
    # ------------------------------------------------------------------

    # BER: bit düzeyinde hata oranı.
    ber = np.mean(bits != rx_bits)

    # SER: en az bir biti hatalı olan sembollerin oranı.
    ser = np.mean(
        np.any(
            bits.reshape(-1, bits_per_symbol) != rx_bits.reshape(-1, bits_per_symbol),
            axis=1,
        )
    )

    return {
        "bits": bits,
        "bit_groups": bit_groups,
        "qam16_symbols": mod_symbols,
        # Geriye dönük uyumluluk için eski anahtarlar korunuyor.
        "bit_pairs": bit_groups,
        "qpsk_symbols": mod_symbols,
        "dd_grid": dd_grid,
        "X_tf": X_tf,
        "x_time_blocks": x_time_blocks,
        "blocks_with_cp": blocks_with_cp,
        "tx_signal": tx_signal,
        "rx_signal": rx_signal,
        "rx_blocks_with_cp": rx_blocks_with_cp,
        "rx_blocks": rx_blocks,
        "Y_tf": Y_tf,
        "Y_dd": Y_dd,
        "rx_symbols": rx_symbols,
        "rx_bits": rx_bits,
        "ber": ber,
        "ser": ser,
        "channel_type": channel_type,
        "snr_db": snr_db,
    }


# ---------------------------------------------------------------------------
# Monte-Carlo SNR süpürmesi
# ---------------------------------------------------------------------------

def run_snr_sweep(channel_type="AWGN", snr_db_list=None, trials_per_snr=50):
    """
    Farklı SNR noktaları için BER ve SER değerlerini Monte-Carlo yöntemiyle hesaplar.

    Parametreler
    ------------
    channel_type : str
        Uygulanacak kanal modeli.
    snr_db_list : list of float, opsiyonel
        Test edilecek SNR değerleri (dB). Verilmezse 0–20 dB, 2 dB adım kullanılır.
    trials_per_snr : int
        Her SNR noktası için bağımsız deneme sayısı.

    Döndürür
    --------
    dict
        snr_db_list, ber_list, ser_list, trials_per_snr ve channel_type.
    """

    if snr_db_list is None:
        snr_db_list = list(range(0, 21, 2))

    if len(snr_db_list) == 0:
        raise ValueError("snr_db_list boş olamaz.")
    if trials_per_snr <= 0:
        raise ValueError("trials_per_snr pozitif bir tam sayı olmalıdır.")

    ber_list = []
    ser_list = []

    for snr_db in snr_db_list:
        # Her SNR noktası için hataları biriktir.
        total_bit_errors = 0
        total_bits = 0
        total_symbol_errors = 0
        total_symbols = 0

        for _ in range(trials_per_snr):
            bits = generate_bits()
            res = run_otfs_simulation(bits=bits, channel_type=channel_type, snr_db=snr_db)

            total_bit_errors += np.sum(res["bits"] != res["rx_bits"])
            total_bits += len(res["bits"])

            tx_sym_bits = res["bits"].reshape(-1, bits_per_symbol)
            rx_sym_bits = res["rx_bits"].reshape(-1, bits_per_symbol)
            total_symbol_errors += np.sum(np.any(tx_sym_bits != rx_sym_bits, axis=1))
            total_symbols += tx_sym_bits.shape[0]

        # Monte-Carlo ortalaması.
        ber_list.append(total_bit_errors / total_bits)
        ser_list.append(total_symbol_errors / total_symbols)

    return {
        "snr_db_list": np.array(snr_db_list),
        "ber_list": np.array(ber_list),
        "ser_list": np.array(ser_list),
        "trials_per_snr": trials_per_snr,
        "channel_type": channel_type,
    }
