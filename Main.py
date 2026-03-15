import numpy as np
import random_bit_generator
from dashboard import launch_otfs_dashboard

# OTFS frame boyutu (M: delay ekseni, N: Doppler ekseni).
M = 4
N = 4
# Her OFDM-benzeri blok icin CP uzunlugu.
cp_len = 1
# 16-QAM -> sembol basina 4 bit.
bits_per_symbol = 4
# Tek bir OTFS frame icin toplam bit sayisi.
num_bits = M * N * bits_per_symbol


def generate_bits():
    # Tek bir frame'i dolduracak kadar rastgele bit uret.
    return np.array(random_bit_generator.generate_random_bits(num_bits))


def qam16_modulation(bits):
    # Modulatorun 4-bit gruplarla calismasi icin boyut kontrolu yap.
    if len(bits) % bits_per_symbol != 0:
        raise ValueError("Bit length must be a multiple of 4 for 16-QAM modulation.")

    bit_quads = bits.reshape(-1, 4)
    symbols = []

    # Gray-benzeri 16-QAM haritasi (ortalama guc normalizasyonu asagida).
    mapping = {
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

    for quad in bit_quads:
        # sqrt(10) ile bolerek 16-QAM ortalama sembol gucunu 1'e cek.
        symbols.append(mapping[tuple(quad)] / np.sqrt(10))

    return bit_quads, np.array(symbols)


def qam16_demodulation(symbols):
    # Her sembol icin 4 bit geri cikartilacak.
    bits_out = []

    def level_to_bits_i(x):
        # In-phase ekseni icin karar esikleri.
        if x < -2 / np.sqrt(10):
            return [0, 0]
        elif x < 0:
            return [0, 1]
        elif x < 2 / np.sqrt(10):
            return [1, 1]
        else:
            return [1, 0]

    def level_to_bits_q(y):
        # Quadrature ekseni icin karar esikleri (ustten alta).
        if y > 2 / np.sqrt(10):
            return [0, 0]
        elif y > 0:
            return [0, 1]
        elif y > -2 / np.sqrt(10):
            return [1, 1]
        else:
            return [1, 0]

    for s in symbols:
        # Haritadaki bit sirasiyla uyumlu olacak sekilde once Q sonra I eklenir.
        q_bits = level_to_bits_q(np.imag(s))
        i_bits = level_to_bits_i(np.real(s))
        bits_out.extend(q_bits + i_bits)

    return np.array(bits_out)


def apply_channel(tx_signal, channel_type="Ideal", snr_db=20.0):
    # Ideal kanalda sinyal birebir gecirilir.
    if channel_type == "Ideal":
        return tx_signal.copy()

    if channel_type == "AWGN":
        # Ortalama sinyal gucunden hedef SNR'a gore gurultu gucunu hesapla.
        signal_power = np.mean(np.abs(tx_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        if snr_linear <= 0 or not np.isfinite(snr_linear):
            raise ValueError("SNR must produce a positive finite linear value.")
        noise_power = signal_power / snr_linear
        # Kompleks AWGN: reel ve imag bilesenlere esit guc dagit.
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*tx_signal.shape) + 1j * np.random.randn(*tx_signal.shape)
        )
        return tx_signal + noise

    raise ValueError(f"Unsupported channel type: {channel_type}")


def run_otfs_simulation(bits, channel_type="Ideal", snr_db=20.0):
    # TX: bit -> 16-QAM sembol.
    bit_groups, mod_symbols = qam16_modulation(bits)

    # Frame boyutu ile sembol sayisi uyumlu mu kontrol et.
    expected_symbols = M * N
    if len(mod_symbols) != expected_symbols:
        raise ValueError(
            f"Expected {expected_symbols} symbols for {M}x{N} grid, got {len(mod_symbols)}."
        )

    # Delay-Doppler gridine yerlestir.
    dd_grid = mod_symbols.reshape(M, N)
    # ISFFT benzeri adimla DD -> TF gecisi.
    X_tf = np.fft.fft(np.fft.ifft(dd_grid, axis=1), axis=0)
    # Heisenberg benzeri adimla TF -> zaman bloklari.
    x_time_blocks = np.fft.ifft(X_tf, axis=0)

    # Her bloga cyclic prefix ekle.
    blocks_with_cp = []
    for n in range(N):
        block = x_time_blocks[:, n]
        cp = block[-cp_len:]
        block_cp = np.concatenate([cp, block])
        blocks_with_cp.append(block_cp)

    # Bloklari kolon-major (Fortran) sirada seri hale getir.
    blocks_with_cp = np.array(blocks_with_cp).T
    tx_signal = blocks_with_cp.flatten(order="F")

    # Kanal etkisi.
    rx_signal = apply_channel(tx_signal, channel_type=channel_type, snr_db=snr_db)

    # RX: seriyi tekrar bloklara ayir, CP'yi at.
    rx_blocks_with_cp = rx_signal.reshape(M + cp_len, N, order="F")
    rx_blocks = rx_blocks_with_cp[cp_len:, :]
    # Wigner/SFFT benzeri adimlarla zaman -> TF -> DD.
    Y_tf = np.fft.fft(rx_blocks, axis=0)
    Y_dd = np.fft.fft(np.fft.ifft(Y_tf, axis=0), axis=1)

    # DD sembollerini tekrar bitlere demodule et.
    rx_symbols = Y_dd.reshape(-1)
    rx_bits = qam16_demodulation(rx_symbols)

    # BER: bit seviyesinde hata orani.
    ber = np.mean(bits != rx_bits)
    # SER: en az bir biti hatali olan sembol orani.
    ser = np.mean(
        np.any(
            bits.reshape(-1, bits_per_symbol) != rx_bits.reshape(-1, bits_per_symbol),
            axis=1
        )
    )

    return {
        "bits": bits,
        # Daha acik adlar.
        "bit_groups": bit_groups,
        "qam16_symbols": mod_symbols,
        # Geriye donuk uyumluluk icin eski adlar korunuyor.
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


def run_snr_sweep(channel_type="AWGN", snr_db_list=None, trials_per_snr=50):
    # Kullanici liste vermezse 0-20 dB araligini 2 dB adimla tara.
    if snr_db_list is None:
        snr_db_list = list(range(0, 21, 2))

    if len(snr_db_list) == 0:
        raise ValueError("snr_db_list must not be empty.")
    if trials_per_snr <= 0:
        raise ValueError("trials_per_snr must be a positive integer.")

    ber_list = []
    ser_list = []

    for snr_db in snr_db_list:
        # Her SNR noktasi icin tum trial hatalarini biriktir.
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

        # Monte Carlo ortalamasi.
        ber_list.append(total_bit_errors / total_bits)
        ser_list.append(total_symbol_errors / total_symbols)

    return {
        "snr_db_list": np.array(snr_db_list),
        "ber_list": np.array(ber_list),
        "ser_list": np.array(ser_list),
        "trials_per_snr": trials_per_snr,
        "channel_type": channel_type,
    }


if __name__ == "__main__":
    launch_otfs_dashboard(run_otfs_simulation, generate_bits, run_snr_sweep)
