import numpy as np
import random_bit_generator
from dashboard import launch_otfs_dashboard

M = 4
N = 4
cp_len = 1
bits_per_symbol = 4
num_bits = M * N * bits_per_symbol


def generate_bits():
    return np.array(random_bit_generator.generate_random_bits(num_bits))


def qam16_modulation(bits):
    bit_quads = bits.reshape(-1, 4)
    symbols = []

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
        symbols.append(mapping[tuple(quad)] / np.sqrt(10))

    return bit_quads, np.array(symbols)


def qam16_demodulation(symbols):
    bits_out = []

    def level_to_bits_i(x):
        if x < -2 / np.sqrt(10):
            return [0, 0]
        elif x < 0:
            return [0, 1]
        elif x < 2 / np.sqrt(10):
            return [1, 1]
        else:
            return [1, 0]

    def level_to_bits_q(y):
        if y > 2 / np.sqrt(10):
            return [0, 0]
        elif y > 0:
            return [0, 1]
        elif y > -2 / np.sqrt(10):
            return [1, 1]
        else:
            return [1, 0]

    for s in symbols:
        q_bits = level_to_bits_q(np.imag(s))
        i_bits = level_to_bits_i(np.real(s))
        bits_out.extend(q_bits + i_bits)

    return np.array(bits_out)


def apply_channel(tx_signal, channel_type="Ideal", snr_db=20.0):
    if channel_type == "Ideal":
        return tx_signal.copy()

    if channel_type == "AWGN":
        signal_power = np.mean(np.abs(tx_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*tx_signal.shape) + 1j * np.random.randn(*tx_signal.shape)
        )
        return tx_signal + noise

    raise ValueError(f"Unsupported channel type: {channel_type}")


def run_otfs_simulation(bits, channel_type="Ideal", snr_db=20.0):
    bit_groups, mod_symbols = qam16_modulation(bits)

    dd_grid = mod_symbols.reshape(M, N)
    X_tf = np.fft.fft(np.fft.ifft(dd_grid, axis=1), axis=0)
    x_time_blocks = np.fft.ifft(X_tf, axis=0)

    blocks_with_cp = []
    for n in range(N):
        block = x_time_blocks[:, n]
        cp = block[-cp_len:]
        block_cp = np.concatenate([cp, block])
        blocks_with_cp.append(block_cp)

    blocks_with_cp = np.array(blocks_with_cp).T
    tx_signal = blocks_with_cp.flatten(order="F")

    rx_signal = apply_channel(tx_signal, channel_type=channel_type, snr_db=snr_db)

    rx_blocks_with_cp = rx_signal.reshape(M + cp_len, N, order="F")
    rx_blocks = rx_blocks_with_cp[cp_len:, :]
    Y_tf = np.fft.fft(rx_blocks, axis=0)
    Y_dd = np.fft.fft(np.fft.ifft(Y_tf, axis=0), axis=1)

    rx_symbols = Y_dd.reshape(-1)
    rx_bits = qam16_demodulation(rx_symbols)

    ber = np.mean(bits != rx_bits)
    ser = np.mean(
        np.any(
            bits.reshape(-1, bits_per_symbol) != rx_bits.reshape(-1, bits_per_symbol),
            axis=1
        )
    )

    return {
        "bits": bits,
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


if __name__ == "__main__":
    launch_otfs_dashboard(run_otfs_simulation, generate_bits)