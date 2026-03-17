# =============================================================================
# channel.py — Kanal modelleri
#   Ideal
#   AWGN
#   Multipath + AWGN
# =============================================================================

import numpy as np


def apply_channel(tx_signal, channel_type="Ideal", snr_db=20.0):
    if channel_type == "Ideal":
        return tx_signal.copy()

    if channel_type == "AWGN":
        signal_power = np.mean(np.abs(tx_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*tx_signal.shape) +
            1j * np.random.randn(*tx_signal.shape)
        )
        return tx_signal + noise

    if channel_type == "Multipath":
        num_paths = 3

        gains = np.random.randn(num_paths) + 1j * np.random.randn(num_paths)
        gains = gains / np.linalg.norm(gains)

        rx_signal = np.convolve(tx_signal, gains, mode="same")

        signal_power = np.mean(np.abs(rx_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*rx_signal.shape) +
            1j * np.random.randn(*rx_signal.shape)
        )
        return rx_signal + noise

    raise ValueError(f"Desteklenmeyen kanal türü: {channel_type}")