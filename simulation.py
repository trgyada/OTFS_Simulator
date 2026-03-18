# =============================================================================
# simulation.py - OTFS TX/RX pipeline + SNR sweep
# =============================================================================

import numpy as np

import random_bit_generator
from config import M, N, cp_len, bits_per_symbol, num_bits
from modulation import qam16_modulation, qam16_demodulation
from channel import apply_channel
from equalizers import apply_equalizer

SUPPORTED_EQUALIZERS = ("ZF", "MMSE")


def _resolve_equalizer(equalization_enabled, equalizer_type):
    if not equalization_enabled:
        return False, "None"

    eq_type = str(equalizer_type).upper()
    if eq_type not in SUPPORTED_EQUALIZERS:
        raise ValueError(f"Desteklenmeyen equalizer tipi: {equalizer_type}")
    return True, eq_type


def generate_bits():
    return np.array(random_bit_generator.generate_random_bits(num_bits))


def run_otfs_simulation(bits, channel_type="Ideal", snr_db=20.0, equalization_enabled=False, equalizer_type="ZF"):
    eq_enabled, eq_type = _resolve_equalizer(equalization_enabled, equalizer_type)
    bits = np.asarray(bits)

    # -----------------------------
    # TX
    # -----------------------------
    bit_groups, tx_symbols = qam16_modulation(bits)

    if len(tx_symbols) != M * N:
        raise ValueError(f"{M}x{N} grid icin {M*N} sembol gerekli.")

    dd_grid = tx_symbols.reshape(M, N)

    # DD -> TF
    X_tf = np.fft.fft(np.fft.ifft(dd_grid, axis=1), axis=0)

    # TF -> time blocks
    x_time_blocks = np.fft.ifft(X_tf, axis=0)

    # CP ekle
    blocks_with_cp = []
    for n_idx in range(N):
        block = x_time_blocks[:, n_idx]
        cp = block[-cp_len:]
        blocks_with_cp.append(np.concatenate([cp, block]))

    blocks_with_cp = np.array(blocks_with_cp).T
    tx_signal = blocks_with_cp.flatten(order="F")

    # -----------------------------
    # CHANNEL
    # -----------------------------
    rx_signal, channel_info = apply_channel(
        tx_signal,
        channel_type=channel_type,
        snr_db=snr_db,
        return_metadata=True,
    )

    # -----------------------------
    # RX
    # -----------------------------
    rx_blocks_with_cp = rx_signal.reshape(M + cp_len, N, order="F")
    rx_blocks = rx_blocks_with_cp[cp_len:, :]

    # time -> TF
    Y_tf_raw = np.fft.fft(rx_blocks, axis=0)

    if eq_enabled:
        # FFT domainunda gürültü gücü M ile ölçeklenir.
        noise_power_tf = float(channel_info.get("noise_power", 0.0)) * M
        Y_tf, equalizer_info = apply_equalizer(
            Y_tf_raw,
            channel_info["impulse_response"],
            method=eq_type,
            noise_power=noise_power_tf,
        )
    else:
        Y_tf = Y_tf_raw
        equalizer_info = {
            "method": "None",
            "channel_response": np.fft.fft(np.array([1.0 + 0.0j]), n=M),
            "weights": np.ones(M, dtype=complex),
            "noise_power": 0.0,
        }

    # TF -> DD
    Y_dd = np.fft.fft(np.fft.ifft(Y_tf, axis=0), axis=1)

    rx_symbols = Y_dd.reshape(-1)
    rx_bits = qam16_demodulation(rx_symbols)

    # -----------------------------
    # METRICS
    # -----------------------------
    ber = np.mean(bits != rx_bits)
    ser = np.mean(
        np.any(
            bits.reshape(-1, bits_per_symbol) != rx_bits.reshape(-1, bits_per_symbol),
            axis=1,
        )
    )

    return {
        "bits": bits,
        "bit_groups": bit_groups,
        "qam16_symbols": tx_symbols,
        # geriye donuk uyum
        "bit_pairs": bit_groups,
        "qpsk_symbols": tx_symbols,
        "dd_grid": dd_grid,
        "X_tf": X_tf,
        "x_time_blocks": x_time_blocks,
        "blocks_with_cp": blocks_with_cp,
        "tx_signal": tx_signal,
        "rx_signal": rx_signal,
        "rx_blocks_with_cp": rx_blocks_with_cp,
        "rx_blocks": rx_blocks,
        "Y_tf_raw": Y_tf_raw,
        "Y_tf_eq": Y_tf,
        "Y_tf": Y_tf,
        "Y_dd": Y_dd,
        "rx_symbols": rx_symbols,
        "rx_bits": rx_bits,
        "ber": ber,
        "ser": ser,
        "channel_type": channel_type,
        "snr_db": snr_db,
        "channel_impulse_response": np.array(channel_info["impulse_response"]),
        "channel_response": np.array(equalizer_info["channel_response"]),
        "equalizer_weights": np.array(equalizer_info["weights"]),
        "equalization_enabled": eq_enabled,
        "equalizer_type": equalizer_info["method"],
        "noise_power": float(channel_info.get("noise_power", 0.0)),
    }


def run_snr_sweep(
    channel_type="AWGN",
    snr_db_list=None,
    trials_per_snr=50,
    equalization_enabled=False,
    equalizer_type="ZF",
):
    eq_enabled, eq_type = _resolve_equalizer(equalization_enabled, equalizer_type)

    if snr_db_list is None:
        snr_db_list = list(range(0, 21, 2))
    if (
        isinstance(trials_per_snr, bool)
        or not isinstance(trials_per_snr, (int, np.integer))
        or trials_per_snr <= 0
    ):
        raise ValueError("trials_per_snr pozitif bir tamsayi olmali.")

    ber_list = []
    ser_list = []

    for snr_db in snr_db_list:
        total_bit_errors = 0
        total_bits = 0
        total_symbol_errors = 0
        total_symbols = 0

        for _ in range(trials_per_snr):
            bits = generate_bits()
            res = run_otfs_simulation(
                bits=bits,
                channel_type=channel_type,
                snr_db=snr_db,
                equalization_enabled=eq_enabled,
                equalizer_type=eq_type,
            )

            total_bit_errors += np.sum(res["bits"] != res["rx_bits"])
            total_bits += len(res["bits"])

            tx_sym_bits = res["bits"].reshape(-1, bits_per_symbol)
            rx_sym_bits = res["rx_bits"].reshape(-1, bits_per_symbol)
            total_symbol_errors += np.sum(np.any(tx_sym_bits != rx_sym_bits, axis=1))
            total_symbols += tx_sym_bits.shape[0]

        ber_list.append(total_bit_errors / total_bits)
        ser_list.append(total_symbol_errors / total_symbols)

    return {
        "snr_db_list": np.array(snr_db_list),
        "ber_list": np.array(ber_list),
        "ser_list": np.array(ser_list),
        "trials_per_snr": trials_per_snr,
        "channel_type": channel_type,
        "equalization_enabled": eq_enabled,
        "equalizer_type": eq_type if eq_enabled else "None",
    }
