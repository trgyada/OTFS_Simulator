import numpy as np


def _to_channel_response(channel_impulse_response, num_subcarriers):
    h = np.asarray(channel_impulse_response, dtype=complex).reshape(-1)
    if h.size == 0:
        raise ValueError("Kanal darbe yaniti bos olamaz.")
    return np.fft.fft(h, n=num_subcarriers)


def apply_equalizer(Y_tf, channel_impulse_response, method="ZF", noise_power=0.0, eps=1e-12):
    if Y_tf.ndim != 2:
        raise ValueError("Y_tf 2-boyutlu olmalidir (subcarrier x block).")

    method_up = method.upper()
    num_subcarriers = Y_tf.shape[0]
    H = _to_channel_response(channel_impulse_response, num_subcarriers)
    H_col = H[:, np.newaxis]

    if method_up == "ZF":
        W = np.zeros_like(H_col, dtype=complex)
        stable = np.abs(H_col) >= eps
        W[stable] = 1.0 / H_col[stable]
    elif method_up == "MMSE":
        noise_power = max(float(noise_power), 0.0)
        denom = (np.abs(H_col) ** 2) + noise_power + eps
        W = np.conj(H_col) / denom
    else:
        raise ValueError(f"Desteklenmeyen equalizer tipi: {method}")

    Y_tf_eq = W * Y_tf
    return Y_tf_eq, {
        "method": method_up,
        "channel_response": H,
        "weights": W[:, 0],
        "noise_power": max(float(noise_power), 0.0),
    }
