# =============================================================================
# channel.py — OTFS simülasyonu için kanal modelleri.
# Desteklenen türler: Ideal (gürültüsüz), AWGN (Additive White Gaussian Noise).
# =============================================================================

import numpy as np


def apply_channel(tx_signal, channel_type="Ideal", snr_db=20.0):
    """
    Verilen TX sinyaline seçilen kanal modelini uygular.

    Parametreler
    ------------
    tx_signal : np.ndarray, karmaşık
        İletilecek zaman-domain sinyal.
    channel_type : str
        "Ideal"  → sinyal değiştirilmeden geçirilir.
        "AWGN"   → belirtilen SNR'a göre karmaşık Gauss gürültüsü eklenir.
    snr_db : float
        AWGN kanalı için sinyal-gürültü oranı (dB cinsinden).

    Döndürür
    --------
    rx_signal : np.ndarray, karmaşık
        Kanal çıkışı sinyal.
    """

    if channel_type == "Ideal":
        # Gürültüsüz geçiş: sinyali olduğu gibi kopyala.
        return tx_signal.copy()

    if channel_type == "AWGN":
        # Ortalama sinyal gücünü hesapla.
        signal_power = np.mean(np.abs(tx_signal) ** 2)

        # dB'yi lineer ölçeğe çevir ve gürültü gücünü belirle.
        snr_linear = 10 ** (snr_db / 10)
        if snr_linear <= 0 or not np.isfinite(snr_linear):
            raise ValueError("SNR değeri pozitif sonlu bir lineer değer üretmelidir.")
        noise_power = signal_power / snr_linear

        # Karmaşık AWGN: gerçek ve sanal bileşenlere eşit güç dağıt.
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*tx_signal.shape)
            + 1j * np.random.randn(*tx_signal.shape)
        )
        return tx_signal + noise

    raise ValueError(f"Desteklenmeyen kanal türü: {channel_type}")
